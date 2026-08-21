output "ecr_repository_url" {
  description = "Container registry URL for the panel segmentation service image."
  value       = aws_ecr_repository.service.repository_url
}

output "model_artifacts_bucket" {
  description = "Private S3 bucket for model checkpoints and sample artifacts."
  value       = aws_s3_bucket.model_artifacts.bucket
}

output "model_read_policy_arn" {
  description = "IAM policy ARN granting read access to model artifacts."
  value       = aws_iam_policy.model_read.arn
}

output "service_account_role_arn" {
  description = "IRSA role ARN, when OIDC inputs are provided."
  value       = try(aws_iam_role.service_account[0].arn, null)
}