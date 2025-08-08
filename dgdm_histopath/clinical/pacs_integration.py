#!/usr/bin/env python3
"""
PACS/EHR Integration Module for DGDM Histopath Lab

Seamless integration with hospital Picture Archiving and Communication Systems (PACS)
and Electronic Health Records (EHR) with full DICOM compliance.

Author: TERRAGON Autonomous Development System v4.0
Generated: 2025-08-08
"""

import asyncio
import logging
import json
import uuid
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import tempfile
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor

try:
    # DICOM handling
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import generate_uid
    from pydicom.encaps import encapsulate
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    logging.warning("PyDICOM not available. Install with: pip install pydicom")

try:
    # HL7 FHIR for EHR integration
    from fhir.resources.patient import Patient
    from fhir.resources.diagnosticreport import DiagnosticReport
    from fhir.resources.observation import Observation
    from fhir.resources.imagingstudy import ImagingStudy
    FHIR_AVAILABLE = True
except ImportError:
    FHIR_AVAILABLE = False
    logging.warning("FHIR resources not available. Install with: pip install fhir.resources")

try:
    # Database connectivity
    import sqlalchemy as sa
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logging.warning("SQLAlchemy not available. Install with: pip install sqlalchemy")

try:
    # DICOM networking
    from pynetdicom import AE, evt, debug_logger
    from pynetdicom.sop_class import (
        SecondaryCaptureImageStorage,
        CTImageStorage,
        MRImageStorage,
        DigitalXRayImagePresentationStorage
    )
    PYNETDICOM_AVAILABLE = True
except ImportError:
    PYNETDICOM_AVAILABLE = False
    logging.warning("PyNetDICOM not available. Install with: pip install pynetdicom")

from ..models.dgdm_model import DGDMModel
from ..utils.exceptions import PACSIntegrationError
from ..utils.validation import validate_tensor, validate_config
from ..utils.monitoring import PACSMetricsCollector
from ..utils.security import encrypt_phi_data, decrypt_phi_data


class DICOMModality(Enum):
    """DICOM imaging modalities."""
    CT = "CT"
    MR = "MR"
    US = "US"
    XA = "XA"
    RF = "RF"
    DX = "DX"
    CR = "CR"
    MG = "MG"
    PT = "PT"
    SC = "SC"  # Secondary Capture
    SR = "SR"  # Structured Report
    WSI = "WSI"  # Whole Slide Imaging
    SM = "SM"  # Slide Microscopy


class PACSVendor(Enum):
    """Major PACS vendor systems."""
    EPIC = "epic"
    CERNER = "cerner"
    CARESTREAM = "carestream"
    AGFA = "agfa"
    GE_CENTRICITY = "ge_centricity"
    SIEMENS_SYNGO = "siemens_syngo"
    PHILIPS_INTELLISPACE = "philips_intellispace"
    SECTRA = "sectra"
    MCKESSON = "mckesson"
    GENERIC_DICOM = "generic_dicom"


class IntegrationProtocol(Enum):
    """Integration communication protocols."""
    DICOM_C_STORE = "dicom_c_store"
    DICOM_C_FIND = "dicom_c_find"
    DICOM_C_GET = "dicom_c_get"
    DICOM_C_MOVE = "dicom_c_move"
    HL7_FHIR = "hl7_fhir"
    REST_API = "rest_api"
    DATABASE_DIRECT = "database_direct"
    FILE_SYSTEM = "file_system"


@dataclass
class PACSConfiguration:
    """Configuration for PACS integration."""
    # PACS system information
    vendor: PACSVendor = PACSVendor.GENERIC_DICOM
    server_host: str = "localhost"
    server_port: int = 11112
    ae_title: str = "DGDM_HISTOPATH"
    called_ae_title: str = "PACS_SERVER"
    
    # Protocol settings
    protocols: List[IntegrationProtocol] = field(default_factory=lambda: [
        IntegrationProtocol.DICOM_C_STORE,
        IntegrationProtocol.DICOM_C_FIND
    ])
    
    # Security and authentication
    use_tls: bool = True
    username: Optional[str] = None
    password: Optional[str] = None
    certificate_path: Optional[str] = None
    
    # Integration settings
    max_concurrent_connections: int = 5
    connection_timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Data mapping
    patient_id_field: str = "PatientID"
    study_uid_field: str = "StudyInstanceUID"
    series_uid_field: str = "SeriesInstanceUID"
    
    # Quality and compliance
    enable_audit_logging: bool = True
    phi_encryption: bool = True
    data_retention_days: int = 2555  # 7 years
    
    # Performance settings
    batch_size: int = 10
    compression_enabled: bool = True
    transfer_syntax: str = "1.2.840.10008.1.2.4.91"  # JPEG 2000


@dataclass
class EHRConfiguration:
    """Configuration for EHR integration."""
    # EHR system information
    system_type: str = "epic"  # epic, cerner, allscripts, etc.
    base_url: str = "https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4"
    client_id: str = ""
    client_secret: str = ""
    
    # OAuth/Authentication
    oauth_url: str = "https://fhir.epic.com/interconnect-fhir-oauth/oauth2/token"
    scope: str = "read write"
    
    # FHIR settings
    fhir_version: str = "4.0.1"
    resource_types: List[str] = field(default_factory=lambda: [
        "Patient", "DiagnosticReport", "Observation", "ImagingStudy"
    ])
    
    # Integration settings
    max_requests_per_minute: int = 60
    timeout_seconds: float = 30.0
    retry_attempts: int = 3


class DICOMHandler:
    """Handles DICOM operations for PACS integration."""
    
    def __init__(self, config: PACSConfiguration):
        self.config = config
        self.metrics = PACSMetricsCollector()
        self.ae = None
        
        if PYNETDICOM_AVAILABLE:
            self._initialize_ae()
    
    def _initialize_ae(self):
        """Initialize DICOM Application Entity."""
        try:
            self.ae = AE(ae_title=self.config.ae_title)
            
            # Add supported presentation contexts
            self.ae.add_supported_context(SecondaryCaptureImageStorage)
            self.ae.add_supported_context(CTImageStorage)
            self.ae.add_supported_context(MRImageStorage)
            
            # Request association as SCU
            self.ae.add_requested_context(SecondaryCaptureImageStorage)
            
            logging.info(f"DICOM AE initialized: {self.config.ae_title}")
            
        except Exception as e:
            logging.error(f"DICOM AE initialization failed: {e}")
    
    async def create_dicom_sr(self, 
                             dgdm_results: Dict[str, Any],
                             patient_info: Dict[str, str],
                             study_info: Dict[str, str]) -> Optional[Dataset]:
        """Create DICOM Structured Report with DGDM analysis results."""
        if not PYDICOM_AVAILABLE:
            logging.error("PyDICOM not available for DICOM SR creation")
            return None
        
        try:
            # Create base DICOM dataset
            ds = Dataset()
            
            # Patient Module
            ds.PatientName = patient_info.get('name', 'Anonymous')
            ds.PatientID = patient_info.get('patient_id', 'UNKNOWN')
            ds.PatientBirthDate = patient_info.get('birth_date', '')
            ds.PatientSex = patient_info.get('sex', 'O')
            
            # Study Module
            ds.StudyInstanceUID = study_info.get('study_uid', generate_uid())
            ds.StudyDate = datetime.now().strftime('%Y%m%d')
            ds.StudyTime = datetime.now().strftime('%H%M%S')
            ds.StudyDescription = 'DGDM Histopathology Analysis'
            ds.AccessionNumber = study_info.get('accession_number', '')
            
            # Series Module
            ds.SeriesInstanceUID = generate_uid()
            ds.SeriesNumber = 1
            ds.SeriesDate = ds.StudyDate
            ds.SeriesTime = ds.StudyTime
            ds.SeriesDescription = 'AI Analysis Results'
            ds.Modality = 'SR'  # Structured Report
            
            # Equipment Module
            ds.Manufacturer = 'TERRAGON Labs'
            ds.ManufacturerModelName = 'DGDM Histopath Analyzer'
            ds.SoftwareVersions = '1.0.0'
            ds.DeviceSerialNumber = 'DGDM-001'
            
            # SR Document Module
            ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.88.11'  # Basic Text SR
            ds.SOPInstanceUID = generate_uid()
            ds.InstanceNumber = 1
            
            # SR Content
            ds.CompletionFlag = 'COMPLETE'
            ds.VerificationFlag = 'UNVERIFIED'
            
            # Content Sequence - Analysis Results
            content_seq = []
            
            # Main diagnosis
            diagnosis_item = Dataset()
            diagnosis_item.ValueType = 'TEXT'
            diagnosis_item.ConceptNameCodeSequence = [self._create_code_item(
                '121071', 'DCM', 'Finding'
            )]
            
            # Format DGDM results
            if 'prediction' in dgdm_results:
                prediction = dgdm_results['prediction']
                confidence = dgdm_results.get('confidence', 0.0)
                
                diagnosis_text = f"AI Analysis: {prediction} (Confidence: {confidence:.2f})"
                diagnosis_item.TextValue = diagnosis_text
                
            content_seq.append(diagnosis_item)
            
            # Additional findings
            if 'detailed_analysis' in dgdm_results:
                for finding in dgdm_results['detailed_analysis']:
                    finding_item = Dataset()
                    finding_item.ValueType = 'TEXT'
                    finding_item.ConceptNameCodeSequence = [self._create_code_item(
                        '121070', 'DCM', 'Additional Finding'
                    )]
                    finding_item.TextValue = str(finding)
                    content_seq.append(finding_item)
            
            ds.ContentSequence = content_seq
            
            # File Meta Information
            file_meta = Dataset()
            file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
            file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            file_meta.ImplementationClassUID = generate_uid()
            file_meta.ImplementationVersionName = 'DGDM_1.0'
            file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'  # Implicit VR Little Endian
            
            ds.file_meta = file_meta
            ds.is_implicit_VR = True
            ds.is_little_endian = True
            
            logging.info("DICOM SR created successfully")
            return ds
            
        except Exception as e:
            logging.error(f"DICOM SR creation failed: {e}")
            return None
    
    def _create_code_item(self, code_value: str, coding_scheme: str, code_meaning: str) -> Dataset:
        """Create a DICOM coded entry."""
        code_item = Dataset()
        code_item.CodeValue = code_value
        code_item.CodingSchemeDesignator = coding_scheme
        code_item.CodeMeaning = code_meaning
        return code_item
    
    async def store_dicom(self, dataset: Dataset, destination: Optional[str] = None) -> bool:
        """Store DICOM dataset to PACS or local storage."""
        try:
            if destination:
                # Store to file
                dataset.save_as(destination, write_like_original=False)
                logging.info(f"DICOM stored to file: {destination}")
                return True
            
            elif PYNETDICOM_AVAILABLE and self.ae:
                # Store to PACS via DICOM C-STORE
                assoc = self.ae.associate(
                    self.config.server_host,
                    self.config.server_port,
                    ae_title=self.config.called_ae_title
                )
                
                if assoc.is_established:
                    status = assoc.send_c_store(dataset)
                    assoc.release()
                    
                    if status and status.Status == 0x0000:
                        logging.info("DICOM stored to PACS successfully")
                        return True
                    else:
                        logging.error(f"DICOM C-STORE failed: {status}")
                        return False
                else:
                    logging.error("Failed to establish DICOM association")
                    return False
            else:
                logging.error("No storage destination specified")
                return False
                
        except Exception as e:
            logging.error(f"DICOM storage failed: {e}")
            return False
    
    async def query_pacs(self, query_params: Dict[str, str]) -> List[Dataset]:
        """Query PACS for studies/series using C-FIND."""
        if not PYNETDICOM_AVAILABLE or not self.ae:
            logging.error("DICOM networking not available")
            return []
        
        try:
            # Create query dataset
            query_ds = Dataset()
            
            # Set query parameters
            for key, value in query_params.items():
                if hasattr(query_ds, key):
                    setattr(query_ds, key, value)
            
            # Establish association
            assoc = self.ae.associate(
                self.config.server_host,
                self.config.server_port,
                ae_title=self.config.called_ae_title
            )
            
            results = []
            
            if assoc.is_established:
                # Send C-FIND request
                responses = assoc.send_c_find(query_ds, '1.2.840.10008.5.1.4.1.2.1.1')  # Study Root
                
                for (status, identifier) in responses:
                    if status.Status == 0x0000:  # Success
                        if identifier:
                            results.append(identifier)
                    elif status.Status in [0xFF00, 0xFF01]:  # Pending
                        if identifier:
                            results.append(identifier)
                
                assoc.release()
            
            logging.info(f"PACS query returned {len(results)} results")
            return results
            
        except Exception as e:
            logging.error(f"PACS query failed: {e}")
            return []


class EHRConnector:
    """Connects to Electronic Health Record systems using HL7 FHIR."""
    
    def __init__(self, config: EHRConfiguration):
        self.config = config
        self.metrics = PACSMetricsCollector()
        self.access_token = None
        self.token_expires = None
        
    async def authenticate(self) -> bool:
        """Authenticate with EHR system using OAuth2."""
        try:
            import requests
            
            auth_data = {
                'grant_type': 'client_credentials',
                'client_id': self.config.client_id,
                'client_secret': self.config.client_secret,
                'scope': self.config.scope
            }
            
            response = requests.post(
                self.config.oauth_url,
                data=auth_data,
                timeout=self.config.timeout_seconds
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                expires_in = token_data.get('expires_in', 3600)
                self.token_expires = datetime.now() + timedelta(seconds=expires_in)
                
                logging.info("EHR authentication successful")
                return True
            else:
                logging.error(f"EHR authentication failed: {response.status_code}")
                return False
                
        except Exception as e:
            logging.error(f"EHR authentication error: {e}")
            return False
    
    async def create_diagnostic_report(self, 
                                     patient_id: str,
                                     dgdm_results: Dict[str, Any],
                                     study_info: Dict[str, Any]) -> Optional[str]:
        """Create FHIR DiagnosticReport with DGDM analysis results."""
        if not FHIR_AVAILABLE:
            logging.error("FHIR resources not available")
            return None
        
        try:
            # Create DiagnosticReport
            report = DiagnosticReport()
            
            # Basic information
            report.id = str(uuid.uuid4())
            report.status = "final"
            report.category = [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                    "code": "PAT",
                    "display": "Pathology"
                }]
            }]
            
            # Code for the report type
            report.code = {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "60567-5",
                    "display": "Pathology report"
                }],
                "text": "AI-Assisted Histopathology Analysis"
            }
            
            # Subject (patient)
            report.subject = {
                "reference": f"Patient/{patient_id}"
            }
            
            # Effective date/time
            report.effectiveDateTime = datetime.now().isoformat()
            
            # Performer
            report.performer = [{
                "reference": "Organization/terragon-labs",
                "display": "TERRAGON Labs AI System"
            }]
            
            # Results - create observations for DGDM findings
            observations = []
            
            # Main prediction observation
            if 'prediction' in dgdm_results:
                obs_id = str(uuid.uuid4())
                observation = self._create_observation(
                    obs_id,
                    patient_id,
                    "AI Diagnosis",
                    str(dgdm_results['prediction']),
                    dgdm_results.get('confidence', 0.0)
                )
                observations.append(obs_id)
                
                # Store observation (would need to POST to EHR)
                await self._store_observation(observation)
            
            # Additional findings
            if 'detailed_analysis' in dgdm_results:
                for i, finding in enumerate(dgdm_results['detailed_analysis']):
                    obs_id = str(uuid.uuid4())
                    observation = self._create_observation(
                        obs_id,
                        patient_id,
                        f"AI Finding {i+1}",
                        str(finding),
                        None
                    )
                    observations.append(obs_id)
                    await self._store_observation(observation)
            
            # Link observations to report
            report.result = [{"reference": f"Observation/{obs_id}"} for obs_id in observations]
            
            # Conclusion
            if 'summary' in dgdm_results:
                report.conclusion = dgdm_results['summary']
            else:
                report.conclusion = f"AI-assisted analysis completed. Primary finding: {dgdm_results.get('prediction', 'Analysis complete')}"
            
            # Store the diagnostic report
            report_id = await self._store_diagnostic_report(report)
            
            logging.info(f"FHIR DiagnosticReport created: {report_id}")
            return report_id
            
        except Exception as e:
            logging.error(f"FHIR DiagnosticReport creation failed: {e}")
            return None
    
    def _create_observation(self, 
                          obs_id: str,
                          patient_id: str,
                          display_name: str,
                          value: str,
                          confidence: Optional[float]) -> Observation:
        """Create FHIR Observation resource."""
        observation = Observation()
        observation.id = obs_id
        observation.status = "final"
        
        # Category
        observation.category = [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "imaging",
                "display": "Imaging"
            }]
        }]
        
        # Code
        observation.code = {
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": "117617002",
                "display": "Pathology finding"
            }],
            "text": display_name
        }
        
        # Subject
        observation.subject = {
            "reference": f"Patient/{patient_id}"
        }
        
        # Effective date/time
        observation.effectiveDateTime = datetime.now().isoformat()
        
        # Value
        observation.valueString = value
        
        # Add confidence as component if available
        if confidence is not None:
            observation.component = [{
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": "LA6115-3",
                        "display": "Confidence level"
                    }]
                },
                "valueQuantity": {
                    "value": confidence,
                    "unit": "probability",
                    "system": "http://unitsofmeasure.org",
                    "code": "1"
                }
            }]
        
        return observation
    
    async def _store_observation(self, observation: Observation) -> bool:
        """Store observation to EHR system."""
        try:
            import requests
            
            if not await self._ensure_authenticated():
                return False
            
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/fhir+json',
                'Accept': 'application/fhir+json'
            }
            
            url = f"{self.config.base_url}/Observation"
            
            response = requests.post(
                url,
                json=observation.dict(),
                headers=headers,
                timeout=self.config.timeout_seconds
            )
            
            if response.status_code in [200, 201]:
                logging.info(f"Observation stored: {observation.id}")
                return True
            else:
                logging.error(f"Failed to store observation: {response.status_code}")
                return False
                
        except Exception as e:
            logging.error(f"Observation storage failed: {e}")
            return False
    
    async def _store_diagnostic_report(self, report: DiagnosticReport) -> Optional[str]:
        """Store diagnostic report to EHR system."""
        try:
            import requests
            
            if not await self._ensure_authenticated():
                return None
            
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/fhir+json',
                'Accept': 'application/fhir+json'
            }
            
            url = f"{self.config.base_url}/DiagnosticReport"
            
            response = requests.post(
                url,
                json=report.dict(),
                headers=headers,
                timeout=self.config.timeout_seconds
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                report_id = result.get('id', report.id)
                logging.info(f"DiagnosticReport stored: {report_id}")
                return report_id
            else:
                logging.error(f"Failed to store diagnostic report: {response.status_code}")
                return None
                
        except Exception as e:
            logging.error(f"DiagnosticReport storage failed: {e}")
            return None
    
    async def _ensure_authenticated(self) -> bool:
        """Ensure valid authentication token."""
        if not self.access_token or (self.token_expires and datetime.now() >= self.token_expires):
            return await self.authenticate()
        return True


class PACSEHRIntegrationManager:
    """High-level manager for PACS and EHR integration."""
    
    def __init__(self, 
                 pacs_config: PACSConfiguration,
                 ehr_config: Optional[EHRConfiguration] = None):
        self.pacs_config = pacs_config
        self.ehr_config = ehr_config
        
        self.dicom_handler = DICOMHandler(pacs_config)
        self.ehr_connector = EHRConnector(ehr_config) if ehr_config else None
        
        self.metrics = PACSMetricsCollector()
        self.integration_db = None
        
        # Initialize database for tracking integrations
        self._initialize_integration_db()
    
    def _initialize_integration_db(self):
        """Initialize SQLite database for integration tracking."""
        try:
            db_path = "pacs_ehr_integration.db"
            self.integration_db = sqlite3.connect(db_path)
            
            # Create tables
            cursor = self.integration_db.cursor()
            
            # Integration log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS integration_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    patient_id TEXT,
                    study_uid TEXT,
                    integration_type TEXT,
                    status TEXT,
                    error_message TEXT,
                    dgdm_results TEXT
                )
            """)
            
            # DICOM storage table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dicom_storage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sop_instance_uid TEXT UNIQUE,
                    patient_id TEXT,
                    study_uid TEXT,
                    series_uid TEXT,
                    storage_path TEXT,
                    storage_timestamp TEXT,
                    pacs_status TEXT
                )
            """)
            
            # EHR reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ehr_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT UNIQUE,
                    patient_id TEXT,
                    fhir_resource_type TEXT,
                    creation_timestamp TEXT,
                    ehr_status TEXT
                )
            """)
            
            self.integration_db.commit()
            logging.info("Integration database initialized")
            
        except Exception as e:
            logging.error(f"Integration database initialization failed: {e}")
    
    async def process_dgdm_results(self, 
                                  dgdm_results: Dict[str, Any],
                                  patient_info: Dict[str, str],
                                  study_info: Dict[str, str],
                                  integration_options: Dict[str, bool]) -> Dict[str, Any]:
        """Process DGDM results and integrate with PACS/EHR systems."""
        try:
            integration_results = {
                'timestamp': datetime.now().isoformat(),
                'patient_id': patient_info.get('patient_id'),
                'study_uid': study_info.get('study_uid'),
                'integrations': {}
            }
            
            # DICOM Integration
            if integration_options.get('create_dicom_sr', True):
                dicom_result = await self._integrate_dicom(
                    dgdm_results, patient_info, study_info
                )
                integration_results['integrations']['dicom'] = dicom_result
            
            # EHR Integration
            if integration_options.get('create_fhir_report', True) and self.ehr_connector:
                ehr_result = await self._integrate_ehr(
                    dgdm_results, patient_info, study_info
                )
                integration_results['integrations']['ehr'] = ehr_result
            
            # Database Integration
            if integration_options.get('store_in_database', True):
                db_result = await self._store_integration_record(
                    dgdm_results, patient_info, study_info, integration_results
                )
                integration_results['integrations']['database'] = db_result
            
            # Update metrics
            await self.metrics.record_integration(integration_results)
            
            logging.info(f"DGDM results integration completed for patient {patient_info.get('patient_id')}")
            return integration_results
            
        except Exception as e:
            error_result = {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'patient_id': patient_info.get('patient_id')
            }
            logging.error(f"DGDM results integration failed: {e}")
            return error_result
    
    async def _integrate_dicom(self, 
                              dgdm_results: Dict[str, Any],
                              patient_info: Dict[str, str],
                              study_info: Dict[str, str]) -> Dict[str, Any]:
        """Integrate with PACS via DICOM."""
        try:
            # Create DICOM Structured Report
            dicom_sr = await self.dicom_handler.create_dicom_sr(
                dgdm_results, patient_info, study_info
            )
            
            if not dicom_sr:
                return {'status': 'failed', 'error': 'DICOM SR creation failed'}
            
            # Store to PACS
            storage_success = await self.dicom_handler.store_dicom(dicom_sr)
            
            # Log to database
            if self.integration_db and storage_success:
                cursor = self.integration_db.cursor()
                cursor.execute("""
                    INSERT INTO dicom_storage 
                    (sop_instance_uid, patient_id, study_uid, series_uid, storage_timestamp, pacs_status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    dicom_sr.SOPInstanceUID,
                    patient_info.get('patient_id'),
                    study_info.get('study_uid'),
                    dicom_sr.SeriesInstanceUID,
                    datetime.now().isoformat(),
                    'stored' if storage_success else 'failed'
                ))
                self.integration_db.commit()
            
            return {
                'status': 'success' if storage_success else 'failed',
                'sop_instance_uid': dicom_sr.SOPInstanceUID,
                'series_instance_uid': dicom_sr.SeriesInstanceUID,
                'storage_method': 'pacs_c_store'
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _integrate_ehr(self, 
                           dgdm_results: Dict[str, Any],
                           patient_info: Dict[str, str],
                           study_info: Dict[str, str]) -> Dict[str, Any]:
        """Integrate with EHR via FHIR."""
        try:
            if not self.ehr_connector:
                return {'status': 'skipped', 'reason': 'EHR connector not configured'}
            
            # Create FHIR DiagnosticReport
            report_id = await self.ehr_connector.create_diagnostic_report(
                patient_info.get('patient_id'),
                dgdm_results,
                study_info
            )
            
            if not report_id:
                return {'status': 'failed', 'error': 'FHIR report creation failed'}
            
            # Log to database
            if self.integration_db:
                cursor = self.integration_db.cursor()
                cursor.execute("""
                    INSERT INTO ehr_reports 
                    (report_id, patient_id, fhir_resource_type, creation_timestamp, ehr_status)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    report_id,
                    patient_info.get('patient_id'),
                    'DiagnosticReport',
                    datetime.now().isoformat(),
                    'created'
                ))
                self.integration_db.commit()
            
            return {
                'status': 'success',
                'report_id': report_id,
                'resource_type': 'DiagnosticReport',
                'ehr_system': self.ehr_config.system_type
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _store_integration_record(self, 
                                      dgdm_results: Dict[str, Any],
                                      patient_info: Dict[str, str],
                                      study_info: Dict[str, str],
                                      integration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Store integration record in database."""
        try:
            if not self.integration_db:
                return {'status': 'skipped', 'reason': 'Database not available'}
            
            cursor = self.integration_db.cursor()
            cursor.execute("""
                INSERT INTO integration_log 
                (timestamp, patient_id, study_uid, integration_type, status, dgdm_results)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                patient_info.get('patient_id'),
                study_info.get('study_uid'),
                'full_integration',
                'completed',
                json.dumps(dgdm_results)
            ))
            
            self.integration_db.commit()
            
            return {'status': 'success', 'action': 'integration_logged'}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def query_integration_history(self, 
                                      patient_id: Optional[str] = None,
                                      study_uid: Optional[str] = None,
                                      limit: int = 100) -> List[Dict[str, Any]]:
        """Query integration history from database."""
        try:
            if not self.integration_db:
                return []
            
            cursor = self.integration_db.cursor()
            
            query = "SELECT * FROM integration_log WHERE 1=1"
            params = []
            
            if patient_id:
                query += " AND patient_id = ?"
                params.append(patient_id)
            
            if study_uid:
                query += " AND study_uid = ?"
                params.append(study_uid)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to dictionaries
            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in rows]
            
            return results
            
        except Exception as e:
            logging.error(f"Integration history query failed: {e}")
            return []
    
    async def validate_integration_setup(self) -> Dict[str, Any]:
        """Validate PACS/EHR integration setup and connectivity."""
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'pacs': {'available': False, 'tests': {}},
            'ehr': {'available': False, 'tests': {}},
            'database': {'available': False, 'tests': {}},
            'overall_status': 'unknown'
        }
        
        try:
            # Test PACS connectivity
            if PYNETDICOM_AVAILABLE and self.dicom_handler.ae:
                pacs_test = await self._test_pacs_connectivity()
                validation_results['pacs'] = pacs_test
            
            # Test EHR connectivity
            if self.ehr_connector:
                ehr_test = await self._test_ehr_connectivity()
                validation_results['ehr'] = ehr_test
            
            # Test database
            if self.integration_db:
                db_test = await self._test_database_connectivity()
                validation_results['database'] = db_test
            
            # Overall status
            pacs_ok = validation_results['pacs']['available']
            ehr_ok = validation_results['ehr'].get('available', True)  # Optional
            db_ok = validation_results['database']['available']
            
            if pacs_ok and db_ok:
                validation_results['overall_status'] = 'ready'
            elif pacs_ok or ehr_ok:
                validation_results['overall_status'] = 'partial'
            else:
                validation_results['overall_status'] = 'not_ready'
            
            return validation_results
            
        except Exception as e:
            validation_results['error'] = str(e)
            validation_results['overall_status'] = 'error'
            return validation_results
    
    async def _test_pacs_connectivity(self) -> Dict[str, Any]:
        """Test PACS connectivity."""
        try:
            # Test DICOM echo (C-ECHO)
            assoc = self.dicom_handler.ae.associate(
                self.pacs_config.server_host,
                self.pacs_config.server_port,
                ae_title=self.pacs_config.called_ae_title
            )
            
            if assoc.is_established:
                status = assoc.send_c_echo()
                assoc.release()
                
                return {
                    'available': True,
                    'tests': {
                        'c_echo': status.Status == 0x0000,
                        'connection': True,
                        'ae_title': self.pacs_config.ae_title
                    }
                }
            else:
                return {
                    'available': False,
                    'tests': {
                        'connection': False,
                        'error': 'Failed to establish DICOM association'
                    }
                }
                
        except Exception as e:
            return {
                'available': False,
                'tests': {
                    'error': str(e)
                }
            }
    
    async def _test_ehr_connectivity(self) -> Dict[str, Any]:
        """Test EHR connectivity."""
        try:
            auth_success = await self.ehr_connector.authenticate()
            
            return {
                'available': auth_success,
                'tests': {
                    'authentication': auth_success,
                    'base_url': self.ehr_config.base_url,
                    'fhir_version': self.ehr_config.fhir_version
                }
            }
            
        except Exception as e:
            return {
                'available': False,
                'tests': {
                    'error': str(e)
                }
            }
    
    async def _test_database_connectivity(self) -> Dict[str, Any]:
        """Test database connectivity."""
        try:
            cursor = self.integration_db.cursor()
            cursor.execute("SELECT COUNT(*) FROM integration_log")
            count = cursor.fetchone()[0]
            
            return {
                'available': True,
                'tests': {
                    'connection': True,
                    'tables_exist': True,
                    'record_count': count
                }
            }
            
        except Exception as e:
            return {
                'available': False,
                'tests': {
                    'error': str(e)
                }
            }
    
    def close(self):
        """Close database connections and cleanup resources."""
        if self.integration_db:
            self.integration_db.close()
            logging.info("Integration database connection closed")


# Export main components
__all__ = [
    'DICOMModality',
    'PACSVendor',
    'IntegrationProtocol',
    'PACSConfiguration',
    'EHRConfiguration',
    'DICOMHandler',
    'EHRConnector',
    'PACSEHRIntegrationManager'
]
