"""
Globalization and Internationalization Framework

Comprehensive global-first implementation with multi-language support,
regional compliance, and cross-cultural medical AI capabilities.
"""

import os
import json
import locale
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import logging
from pathlib import Path

try:
    import babel
    from babel import Locale, dates, numbers, units
    from babel.support import Translations
    BABEL_AVAILABLE = True
except ImportError:
    BABEL_AVAILABLE = False

from dgdm_histopath.utils.exceptions import DGDMException


class SupportedLanguage(Enum):
    """Supported languages for global deployment."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh_CN"
    CHINESE_TRADITIONAL = "zh_TW"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    DUTCH = "nl"
    ARABIC = "ar"
    HINDI = "hi"
    RUSSIAN = "ru"


class RegionalCompliance(Enum):
    """Regional compliance frameworks."""
    GDPR_EU = "gdpr_eu"           # European Union
    HIPAA_US = "hipaa_us"         # United States
    PIPEDA_CA = "pipeda_ca"       # Canada
    PDPA_SG = "pdpa_sg"          # Singapore
    LGPD_BR = "lgpd_br"          # Brazil
    PIPL_CN = "pipl_cn"          # China
    APPI_JP = "appi_jp"          # Japan
    PIPA_KR = "pipa_kr"          # South Korea
    DPA_UK = "dpa_uk"            # United Kingdom
    CCPA_CA_US = "ccpa_ca_us"    # California, US


class MedicalStandard(Enum):
    """International medical standards."""
    ICD_10 = "icd_10"            # International Classification of Diseases
    ICD_11 = "icd_11"            # Latest ICD version
    SNOMED_CT = "snomed_ct"      # Systematized Nomenclature of Medicine
    LOINC = "loinc"              # Logical Observation Identifiers
    HL7_FHIR = "hl7_fhir"        # Healthcare data exchange
    DICOM = "dicom"              # Medical imaging standard
    ISO_13485 = "iso_13485"      # Medical device quality
    FDA_510K = "fda_510k"        # US FDA premarket notification
    CE_MDR = "ce_mdr"            # European medical device regulation
    PMDA_JP = "pmda_jp"          # Japan pharmaceutical approval


@dataclass
class LocalizationConfig:
    """Configuration for localization."""
    language: SupportedLanguage
    region: str
    timezone: str
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    number_format: str = "decimal"
    currency: str = "USD"
    compliance_frameworks: List[RegionalCompliance] = field(default_factory=list)
    medical_standards: List[MedicalStandard] = field(default_factory=list)


@dataclass
class GlobalizedContent:
    """Globalized content container."""
    content_id: str
    translations: Dict[str, str] = field(default_factory=dict)
    regional_variations: Dict[str, Dict[str, str]] = field(default_factory=dict)
    medical_terminology: Dict[str, str] = field(default_factory=dict)
    compliance_notes: Dict[str, str] = field(default_factory=dict)


class InternationalizationManager:
    """
    Comprehensive internationalization manager for medical AI systems
    with clinical terminology support and regulatory compliance.
    """
    
    def __init__(
        self,
        default_language: SupportedLanguage = SupportedLanguage.ENGLISH,
        translations_directory: Optional[str] = None,
        enable_clinical_terminology: bool = True
    ):
        self.default_language = default_language
        self.translations_directory = translations_directory or "locales"
        self.enable_clinical_terminology = enable_clinical_terminology
        
        # Localization state
        self.current_config = None
        self.translations = {}
        self.clinical_terms = {}
        self.regional_configurations = {}
        
        # Medical terminology mappings
        self.medical_terminology_maps = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize default configuration
        self._initialize_default_configs()
        self._load_translations()
        
        if enable_clinical_terminology:
            self._load_clinical_terminology()
    
    def _initialize_default_configs(self):
        """Initialize default regional configurations."""
        # United States configuration
        self.regional_configurations["US"] = LocalizationConfig(
            language=SupportedLanguage.ENGLISH,
            region="US",
            timezone="America/New_York",
            date_format="%m/%d/%Y",
            time_format="%I:%M %p",
            currency="USD",
            compliance_frameworks=[RegionalCompliance.HIPAA_US, RegionalCompliance.CCPA_CA_US],
            medical_standards=[MedicalStandard.ICD_10, MedicalStandard.SNOMED_CT, MedicalStandard.LOINC, MedicalStandard.FDA_510K]
        )
        
        # European Union configuration
        self.regional_configurations["EU"] = LocalizationConfig(
            language=SupportedLanguage.ENGLISH,
            region="EU",
            timezone="Europe/Brussels",
            date_format="%d.%m.%Y",
            time_format="%H:%M",
            currency="EUR",
            compliance_frameworks=[RegionalCompliance.GDPR_EU, RegionalCompliance.CE_MDR],
            medical_standards=[MedicalStandard.ICD_11, MedicalStandard.SNOMED_CT, MedicalStandard.DICOM]
        )
        
        # Japan configuration
        self.regional_configurations["JP"] = LocalizationConfig(
            language=SupportedLanguage.JAPANESE,
            region="JP",
            timezone="Asia/Tokyo",
            date_format="%Y年%m月%d日",
            time_format="%H時%M分",
            currency="JPY",
            compliance_frameworks=[RegionalCompliance.APPI_JP],
            medical_standards=[MedicalStandard.ICD_10, MedicalStandard.PMDA_JP]
        )
        
        # China configuration
        self.regional_configurations["CN"] = LocalizationConfig(
            language=SupportedLanguage.CHINESE_SIMPLIFIED,
            region="CN",
            timezone="Asia/Shanghai",
            date_format="%Y年%m月%d日",
            time_format="%H:%M",
            currency="CNY",
            compliance_frameworks=[RegionalCompliance.PIPL_CN],
            medical_standards=[MedicalStandard.ICD_10, MedicalStandard.DICOM]
        )
        
        # Brazil configuration
        self.regional_configurations["BR"] = LocalizationConfig(
            language=SupportedLanguage.PORTUGUESE,
            region="BR",
            timezone="America/Sao_Paulo",
            date_format="%d/%m/%Y",
            time_format="%H:%M",
            currency="BRL",
            compliance_frameworks=[RegionalCompliance.LGPD_BR],
            medical_standards=[MedicalStandard.ICD_10, MedicalStandard.SNOMED_CT]
        )
    
    def _load_translations(self):
        """Load translation files for all supported languages."""
        # Default translations (embedded)
        self.translations = {
            SupportedLanguage.ENGLISH.value: {
                "app_name": "DGDM Histopath Lab",
                "model_inference": "Model Inference",
                "analysis_complete": "Analysis Complete",
                "diagnosis": "Diagnosis",
                "confidence": "Confidence",
                "recommendations": "Recommendations",
                "patient_data": "Patient Data",
                "medical_image": "Medical Image",
                "processing": "Processing...",
                "error_occurred": "An error occurred",
                "unauthorized": "Unauthorized access",
                "data_privacy": "Data Privacy",
                "clinical_validation": "Clinical Validation",
                "quality_assurance": "Quality Assurance",
                "performance_metrics": "Performance Metrics",
                "regulatory_compliance": "Regulatory Compliance"
            },
            SupportedLanguage.SPANISH.value: {
                "app_name": "Laboratorio DGDM Histopatología",
                "model_inference": "Inferencia del Modelo",
                "analysis_complete": "Análisis Completado",
                "diagnosis": "Diagnóstico",
                "confidence": "Confianza",
                "recommendations": "Recomendaciones",
                "patient_data": "Datos del Paciente",
                "medical_image": "Imagen Médica",
                "processing": "Procesando...",
                "error_occurred": "Ocurrió un error",
                "unauthorized": "Acceso no autorizado",
                "data_privacy": "Privacidad de Datos",
                "clinical_validation": "Validación Clínica",
                "quality_assurance": "Aseguramiento de Calidad",
                "performance_metrics": "Métricas de Rendimiento",
                "regulatory_compliance": "Cumplimiento Regulatorio"
            },
            SupportedLanguage.FRENCH.value: {
                "app_name": "Laboratoire DGDM Histopathologie",
                "model_inference": "Inférence du Modèle",
                "analysis_complete": "Analyse Terminée",
                "diagnosis": "Diagnostic",
                "confidence": "Confiance",
                "recommendations": "Recommandations",
                "patient_data": "Données Patient",
                "medical_image": "Image Médicale",
                "processing": "Traitement en cours...",
                "error_occurred": "Une erreur s'est produite",
                "unauthorized": "Accès non autorisé",
                "data_privacy": "Confidentialité des Données",
                "clinical_validation": "Validation Clinique",
                "quality_assurance": "Assurance Qualité",
                "performance_metrics": "Métriques de Performance",
                "regulatory_compliance": "Conformité Réglementaire"
            },
            SupportedLanguage.GERMAN.value: {
                "app_name": "DGDM Histopathologie Labor",
                "model_inference": "Modell-Inferenz",
                "analysis_complete": "Analyse Abgeschlossen",
                "diagnosis": "Diagnose",
                "confidence": "Vertrauen",
                "recommendations": "Empfehlungen",
                "patient_data": "Patientendaten",
                "medical_image": "Medizinisches Bild",
                "processing": "Verarbeitung...",
                "error_occurred": "Ein Fehler ist aufgetreten",
                "unauthorized": "Unbefugter Zugriff",
                "data_privacy": "Datenschutz",
                "clinical_validation": "Klinische Validierung",
                "quality_assurance": "Qualitätssicherung",
                "performance_metrics": "Leistungsmetriken",
                "regulatory_compliance": "Regulatorische Compliance"
            },
            SupportedLanguage.JAPANESE.value: {
                "app_name": "DGDM病理学研究室",
                "model_inference": "モデル推論",
                "analysis_complete": "分析完了",
                "diagnosis": "診断",
                "confidence": "信頼度",
                "recommendations": "推奨事項",
                "patient_data": "患者データ",
                "medical_image": "医療画像",
                "processing": "処理中...",
                "error_occurred": "エラーが発生しました",
                "unauthorized": "不正アクセス",
                "data_privacy": "データプライバシー",
                "clinical_validation": "臨床検証",
                "quality_assurance": "品質保証",
                "performance_metrics": "パフォーマンス指標",
                "regulatory_compliance": "規制コンプライアンス"
            },
            SupportedLanguage.CHINESE_SIMPLIFIED.value: {
                "app_name": "DGDM组织病理学实验室",
                "model_inference": "模型推理",
                "analysis_complete": "分析完成",
                "diagnosis": "诊断",
                "confidence": "置信度",
                "recommendations": "建议",
                "patient_data": "患者数据",
                "medical_image": "医学图像",
                "processing": "处理中...",
                "error_occurred": "发生错误",
                "unauthorized": "未授权访问",
                "data_privacy": "数据隐私",
                "clinical_validation": "临床验证",
                "quality_assurance": "质量保证",
                "performance_metrics": "性能指标",
                "regulatory_compliance": "监管合规"
            }
        }
    
    def _load_clinical_terminology(self):
        """Load clinical terminology mappings."""
        # ICD-10 common codes with translations
        self.clinical_terms = {
            "malignant_neoplasm": {
                SupportedLanguage.ENGLISH.value: "Malignant neoplasm",
                SupportedLanguage.SPANISH.value: "Neoplasia maligna",
                SupportedLanguage.FRENCH.value: "Néoplasme malin",
                SupportedLanguage.GERMAN.value: "Bösartige Neubildung",
                SupportedLanguage.JAPANESE.value: "悪性新生物",
                SupportedLanguage.CHINESE_SIMPLIFIED.value: "恶性肿瘤"
            },
            "benign_neoplasm": {
                SupportedLanguage.ENGLISH.value: "Benign neoplasm",
                SupportedLanguage.SPANISH.value: "Neoplasia benigna",
                SupportedLanguage.FRENCH.value: "Néoplasme bénin",
                SupportedLanguage.GERMAN.value: "Gutartige Neubildung",
                SupportedLanguage.JAPANESE.value: "良性新生物",
                SupportedLanguage.CHINESE_SIMPLIFIED.value: "良性肿瘤"
            },
            "inflammatory_process": {
                SupportedLanguage.ENGLISH.value: "Inflammatory process",
                SupportedLanguage.SPANISH.value: "Proceso inflamatorio",
                SupportedLanguage.FRENCH.value: "Processus inflammatoire",
                SupportedLanguage.GERMAN.value: "Entzündungsprozess",
                SupportedLanguage.JAPANESE.value: "炎症過程",
                SupportedLanguage.CHINESE_SIMPLIFIED.value: "炎症过程"
            },
            "tissue_necrosis": {
                SupportedLanguage.ENGLISH.value: "Tissue necrosis",
                SupportedLanguage.SPANISH.value: "Necrosis tisular",
                SupportedLanguage.FRENCH.value: "Nécrose tissulaire",
                SupportedLanguage.GERMAN.value: "Gewebenekrose",
                SupportedLanguage.JAPANESE.value: "組織壊死",
                SupportedLanguage.CHINESE_SIMPLIFIED.value: "组织坏死"
            }
        }
    
    def set_localization(self, language: SupportedLanguage, region: Optional[str] = None):
        """Set current localization configuration."""
        if region and region in self.regional_configurations:
            self.current_config = self.regional_configurations[region]
            # Override language if specified
            if language != self.current_config.language:
                self.current_config.language = language
        else:
            # Create basic configuration
            self.current_config = LocalizationConfig(
                language=language,
                region=region or "GLOBAL",
                timezone="UTC",
                compliance_frameworks=[],
                medical_standards=[MedicalStandard.ICD_10]
            )
        
        self.logger.info(f"Localization set to: {language.value} ({region or 'GLOBAL'})")
    
    def translate(self, key: str, language: Optional[SupportedLanguage] = None) -> str:
        """Translate text to specified language."""
        target_language = language or (self.current_config.language if self.current_config else self.default_language)
        
        # Get translation
        lang_code = target_language.value
        if lang_code in self.translations and key in self.translations[lang_code]:
            return self.translations[lang_code][key]
        
        # Fallback to English
        if SupportedLanguage.ENGLISH.value in self.translations and key in self.translations[SupportedLanguage.ENGLISH.value]:
            return self.translations[SupportedLanguage.ENGLISH.value][key]
        
        # Return key if no translation found
        return key
    
    def translate_clinical_term(self, term_key: str, language: Optional[SupportedLanguage] = None) -> str:
        """Translate clinical terminology."""
        if not self.enable_clinical_terminology:
            return term_key
        
        target_language = language or (self.current_config.language if self.current_config else self.default_language)
        lang_code = target_language.value
        
        if term_key in self.clinical_terms and lang_code in self.clinical_terms[term_key]:
            return self.clinical_terms[term_key][lang_code]
        
        # Fallback to English clinical term
        if term_key in self.clinical_terms and SupportedLanguage.ENGLISH.value in self.clinical_terms[term_key]:
            return self.clinical_terms[term_key][SupportedLanguage.ENGLISH.value]
        
        return term_key
    
    def format_datetime(self, dt: datetime, include_time: bool = True) -> str:
        """Format datetime according to localization settings."""
        if not self.current_config:
            return dt.isoformat()
        
        try:
            if BABEL_AVAILABLE:
                locale_obj = Locale(self.current_config.language.value)
                if include_time:
                    return dates.format_datetime(dt, locale=locale_obj)
                else:
                    return dates.format_date(dt.date(), locale=locale_obj)
            else:
                # Fallback formatting
                if include_time:
                    return dt.strftime(f"{self.current_config.date_format} {self.current_config.time_format}")
                else:
                    return dt.strftime(self.current_config.date_format)
        except:
            return dt.isoformat()
    
    def format_number(self, number: Union[int, float], decimal_places: Optional[int] = None) -> str:
        """Format number according to localization settings."""
        if not self.current_config:
            return str(number)
        
        try:
            if BABEL_AVAILABLE:
                locale_obj = Locale(self.current_config.language.value)
                if isinstance(number, float):
                    return numbers.format_decimal(number, locale=locale_obj)
                else:
                    return numbers.format_decimal(float(number), locale=locale_obj)
            else:
                # Basic formatting
                if decimal_places is not None:
                    return f"{number:.{decimal_places}f}"
                return str(number)
        except:
            return str(number)
    
    def get_compliance_requirements(self, region: Optional[str] = None) -> List[RegionalCompliance]:
        """Get compliance requirements for region."""
        config = self.current_config
        if region and region in self.regional_configurations:
            config = self.regional_configurations[region]
        
        return config.compliance_frameworks if config else []
    
    def get_medical_standards(self, region: Optional[str] = None) -> List[MedicalStandard]:
        """Get applicable medical standards for region."""
        config = self.current_config
        if region and region in self.regional_configurations:
            config = self.regional_configurations[region]
        
        return config.medical_standards if config else [MedicalStandard.ICD_10]
    
    def validate_regional_compliance(self, data: Dict[str, Any], region: str) -> Dict[str, bool]:
        """Validate data against regional compliance requirements."""
        if region not in self.regional_configurations:
            return {"unknown_region": False}
        
        config = self.regional_configurations[region]
        compliance_results = {}
        
        for framework in config.compliance_frameworks:
            if framework == RegionalCompliance.GDPR_EU:
                compliance_results["gdpr_consent"] = "user_consent" in data
                compliance_results["gdpr_data_minimization"] = len(data.get("personal_data", {})) <= 10
                compliance_results["gdpr_purpose_limitation"] = "data_purpose" in data
            
            elif framework == RegionalCompliance.HIPAA_US:
                compliance_results["hipaa_phi_protection"] = "phi_removed" in data and data["phi_removed"]
                compliance_results["hipaa_access_controls"] = "access_log" in data
                compliance_results["hipaa_audit_trail"] = "audit_trail" in data
            
            elif framework == RegionalCompliance.PIPL_CN:
                compliance_results["pipl_consent"] = "explicit_consent" in data
                compliance_results["pipl_data_localization"] = data.get("data_stored_in_china", False)
            
            # Add more compliance checks as needed
        
        return compliance_results
    
    def create_localized_report(
        self,
        report_data: Dict[str, Any],
        language: Optional[SupportedLanguage] = None,
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a localized medical report."""
        target_language = language or (self.current_config.language if self.current_config else self.default_language)
        
        localized_report = {
            "report_metadata": {
                "language": target_language.value,
                "region": region or (self.current_config.region if self.current_config else "GLOBAL"),
                "generated_at": self.format_datetime(datetime.now()),
                "compliance_frameworks": [f.value for f in self.get_compliance_requirements(region)],
                "medical_standards": [s.value for s in self.get_medical_standards(region)]
            }
        }
        
        # Translate standard report sections
        sections_to_translate = [
            "diagnosis", "recommendations", "clinical_findings",
            "quality_assurance", "confidence_metrics"
        ]
        
        for section in sections_to_translate:
            if section in report_data:
                localized_report[self.translate(section, target_language)] = report_data[section]
        
        # Translate clinical terminology
        if "clinical_terms" in report_data:
            localized_clinical_terms = {}
            for term_key, term_data in report_data["clinical_terms"].items():
                translated_term = self.translate_clinical_term(term_key, target_language)
                localized_clinical_terms[translated_term] = term_data
            localized_report[self.translate("clinical_findings", target_language)] = localized_clinical_terms
        
        # Add compliance validation if region specified
        if region:
            compliance_results = self.validate_regional_compliance(report_data, region)
            localized_report["compliance_validation"] = compliance_results
        
        return localized_report


class GlobalDeploymentManager:
    """
    Manages global deployment with regional configurations,
    compliance validation, and cultural adaptations.
    """
    
    def __init__(self):
        self.i18n_manager = InternationalizationManager()
        self.regional_deployments = {}
        self.compliance_validators = {}
        
        self.logger = logging.getLogger(__name__)
    
    def register_regional_deployment(
        self,
        region: str,
        deployment_config: Dict[str, Any],
        compliance_requirements: List[RegionalCompliance]
    ):
        """Register a regional deployment configuration."""
        self.regional_deployments[region] = {
            "config": deployment_config,
            "compliance": compliance_requirements,
            "registered_at": datetime.now(),
            "status": "registered"
        }
        
        self.logger.info(f"Registered regional deployment: {region}")
    
    def validate_global_deployment(self) -> Dict[str, Any]:
        """Validate global deployment readiness."""
        validation_results = {
            "regions_configured": len(self.regional_deployments),
            "languages_supported": len(self.i18n_manager.translations),
            "compliance_frameworks": len(set().union(*[
                dep["compliance"] for dep in self.regional_deployments.values()
            ])) if self.regional_deployments else 0,
            "regional_validations": {}
        }
        
        for region, deployment in self.regional_deployments.items():
            region_validation = {
                "deployment_ready": "config" in deployment and deployment["config"],
                "compliance_configured": len(deployment.get("compliance", [])) > 0,
                "localization_ready": region in self.i18n_manager.regional_configurations
            }
            validation_results["regional_validations"][region] = region_validation
        
        # Overall readiness
        validation_results["global_ready"] = (
            validation_results["regions_configured"] >= 3 and
            validation_results["languages_supported"] >= 5 and
            validation_results["compliance_frameworks"] >= 3
        )
        
        return validation_results
    
    def get_deployment_recommendations(self) -> List[str]:
        """Get recommendations for improving global deployment."""
        recommendations = []
        
        validation = self.validate_global_deployment()
        
        if validation["regions_configured"] < 5:
            recommendations.append("Add more regional configurations for better global coverage")
        
        if validation["languages_supported"] < 10:
            recommendations.append("Expand language support for broader accessibility")
        
        if validation["compliance_frameworks"] < 5:
            recommendations.append("Add support for more compliance frameworks")
        
        # Check for missing major regions
        major_regions = ["US", "EU", "JP", "CN", "BR", "IN", "CA", "AU"]
        missing_regions = [r for r in major_regions if r not in self.regional_deployments]
        
        if missing_regions:
            recommendations.append(f"Consider adding support for major regions: {', '.join(missing_regions)}")
        
        return recommendations


# Global instance for easy access
_global_i18n_manager = None


def get_i18n_manager() -> InternationalizationManager:
    """Get global internationalization manager instance."""
    global _global_i18n_manager
    if _global_i18n_manager is None:
        _global_i18n_manager = InternationalizationManager()
    return _global_i18n_manager


def translate(key: str, language: Optional[SupportedLanguage] = None) -> str:
    """Convenience function for translation."""
    return get_i18n_manager().translate(key, language)


def translate_clinical(term: str, language: Optional[SupportedLanguage] = None) -> str:
    """Convenience function for clinical term translation."""
    return get_i18n_manager().translate_clinical_term(term, language)


# Example usage and testing
if __name__ == "__main__":
    print("Globalization Framework Loaded")
    print("Global capabilities:")
    print("- Multi-language support with clinical terminology")
    print("- Regional compliance validation (GDPR, HIPAA, PIPL, etc.)")
    print("- Cultural adaptation for medical AI")
    print("- International medical standards support")
    print("- Comprehensive localization management")