variable "aws_region" {
  description = "AWS region for deployment infrastructure."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Short project name used in resource names."
  type        = string
  default     = "panel-seg"
}

variable "environment" {
  description = "Deployment environment name."
  type        = string
  default     = "dev"
}

variable "eks_oidc_provider_arn" {
  description = "Existing EKS cluster OIDC provider ARN for IRSA. Leave empty to skip service-account role creation."
  type        = string
  default     = ""
}

variable "eks_oidc_provider_url" {
  description = "Existing EKS OIDC issuer URL without https://, for example oidc.eks.us-east-1.amazonaws.com/id/ABC."
  type        = string
  default     = ""
}

variable "kubernetes_namespace" {
  description = "Kubernetes namespace used by the service."
  type        = string
  default     = "panel-seg"
}

variable "kubernetes_service_account" {
  description = "Kubernetes service account that may read model artifacts."
  type        = string
  default     = "panel-seg-api"
}