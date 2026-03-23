"""
AWS Infrastructure — Tourist Prediction MLOps
Deploys to AWS using ECS Fargate + ALB + ECR + S3 + CloudWatch

Architecture:
  Route53 → ALB → ECS Fargate (API) → S3 (models) → CloudWatch (logs)
  
Prerequisites:
  pip install aws-cdk-lib constructs boto3
  aws configure (set credentials)
  cdk bootstrap aws://ACCOUNT_ID/REGION
"""

import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_ecr as ecr,
    aws_s3 as s3,
    aws_iam as iam,
    aws_logs as logs,
    aws_cloudwatch as cloudwatch,
    aws_secretsmanager as secretsmanager,
    Duration, RemovalPolicy, CfnOutput
)
from constructs import Construct


class TourismMLOpsStack(Stack):
    """
    Complete AWS infrastructure for Tourist Prediction MLOps platform.
    
    Components:
    - VPC with public/private subnets
    - ECR repository for Docker images
    - ECS Fargate cluster + service
    - Application Load Balancer
    - S3 bucket for model artifacts + MLflow
    - CloudWatch dashboards + alarms
    - IAM roles + policies
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        env_name = self.node.try_get_context("env") or "prod"
        app_name = "tourist-prediction"

        # ── VPC ────────────────────────────────────────────────────
        vpc = ec2.Vpc(
            self, "TourismVPC",
            vpc_name=f"{app_name}-vpc-{env_name}",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24
                )
            ]
        )

        # ── ECR Repository ─────────────────────────────────────────
        ecr_repo = ecr.Repository(
            self, "TourismECR",
            repository_name=f"{app_name}-api",
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    max_image_count=10,
                    description="Keep last 10 images"
                )
            ]
        )

        # ── S3 Buckets ─────────────────────────────────────────────
        model_bucket = s3.Bucket(
            self, "ModelArtifactsBucket",
            bucket_name=f"{app_name}-models-{self.account}",
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="archive-old-models",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INTELLIGENT_TIERING,
                            transition_after=Duration.days(90)
                        )
                    ]
                )
            ]
        )

        mlflow_bucket = s3.Bucket(
            self, "MLflowBucket",
            bucket_name=f"{app_name}-mlflow-{self.account}",
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL
        )

        # ── ECS Cluster ────────────────────────────────────────────
        cluster = ecs.Cluster(
            self, "TourismCluster",
            cluster_name=f"{app_name}-cluster-{env_name}",
            vpc=vpc,
            container_insights=True
        )

        # ── IAM Role for ECS Tasks ─────────────────────────────────
        task_role = iam.Role(
            self, "ECSTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            role_name=f"{app_name}-task-role-{env_name}",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                )
            ]
        )

        # Grant S3 access to task
        model_bucket.grant_read_write(task_role)
        mlflow_bucket.grant_read_write(task_role)

        # CloudWatch logs permission
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                resources=["*"]
            )
        )

        # ── CloudWatch Log Group ───────────────────────────────────
        log_group = logs.LogGroup(
            self, "APILogGroup",
            log_group_name=f"/ecs/{app_name}/{env_name}",
            removal_policy=RemovalPolicy.DESTROY,
            retention=logs.RetentionDays.ONE_MONTH
        )

        # ── ECS Task Definition ────────────────────────────────────
        task_definition = ecs.FargateTaskDefinition(
            self, "TourismTaskDef",
            family=f"{app_name}-task-{env_name}",
            cpu=1024,          # 1 vCPU
            memory_limit_mib=2048,  # 2 GB
            task_role=task_role
        )

        container = task_definition.add_container(
            "TourismAPIContainer",
            container_name=f"{app_name}-api",
            image=ecs.ContainerImage.from_ecr_repository(ecr_repo, tag="latest"),
            port_mappings=[
                ecs.PortMapping(container_port=8080, protocol=ecs.Protocol.TCP)
            ],
            environment={
                "PYTHONPATH": "/app",
                "MODEL_BUCKET": model_bucket.bucket_name,
                "MLFLOW_BUCKET": mlflow_bucket.bucket_name,
                "ENV": env_name,
                "LOG_LEVEL": "INFO"
            },
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix=f"{app_name}-api",
                log_group=log_group
            ),
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL",
                         "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8080/health')\""],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(10),
                retries=3,
                start_period=Duration.seconds(60)
            )
        )

        # ── Fargate Service + ALB ──────────────────────────────────
        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self, "TourismFargateService",
            service_name=f"{app_name}-service-{env_name}",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=2,
            public_load_balancer=True,
            assign_public_ip=False,
            listener_port=80,
        )

        # ── Auto Scaling ───────────────────────────────────────────
        scalable_target = fargate_service.service.auto_scale_task_count(
            min_capacity=2,
            max_capacity=10
        )

        scalable_target.scale_on_cpu_utilization(
            "CPUScaling",
            target_utilization_percent=70,
            scale_in_cooldown=Duration.seconds(300),
            scale_out_cooldown=Duration.seconds(60)
        )

        scalable_target.scale_on_request_count(
            "RequestScaling",
            requests_per_target=100,
            target_group=fargate_service.target_group
        )

        # ALB Health Check
        fargate_service.target_group.configure_health_check(
            path="/health",
            healthy_http_codes="200",
            interval=Duration.seconds(30),
            timeout=Duration.seconds(10),
            healthy_threshold_count=2,
            unhealthy_threshold_count=3
        )

        # ── CloudWatch Dashboard ───────────────────────────────────
        dashboard = cloudwatch.Dashboard(
            self, "MLOpsDashboard",
            dashboard_name=f"{app_name}-dashboard-{env_name}"
        )

        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="ECS CPU Utilization",
                left=[
                    fargate_service.service.metric_cpu_utilization(
                        period=Duration.minutes(1)
                    )
                ],
                width=12
            ),
            cloudwatch.GraphWidget(
                title="ECS Memory Utilization",
                left=[
                    fargate_service.service.metric_memory_utilization(
                        period=Duration.minutes(1)
                    )
                ],
                width=12
            )
        )

        # ── Outputs ────────────────────────────────────────────────
        CfnOutput(self, "LoadBalancerDNS",
                  value=fargate_service.load_balancer.load_balancer_dns_name,
                  description="API endpoint URL",
                  export_name=f"{app_name}-alb-dns")

        CfnOutput(self, "ECRRepository",
                  value=ecr_repo.repository_uri,
                  description="ECR repository URI",
                  export_name=f"{app_name}-ecr-uri")

        CfnOutput(self, "ModelBucket",
                  value=model_bucket.bucket_name,
                  description="S3 bucket for model artifacts",
                  export_name=f"{app_name}-model-bucket")

        CfnOutput(self, "MLflowBucket",
                  value=mlflow_bucket.bucket_name,
                  description="S3 bucket for MLflow tracking",
                  export_name=f"{app_name}-mlflow-bucket")


# ─────────────────────────────────────────────────────────────────
# CDK App Entry Point
# ─────────────────────────────────────────────────────────────────

app = cdk.App()

TourismMLOpsStack(
    app, "TourismMLOpsStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account") or "YOUR_AWS_ACCOUNT_ID",
        region=app.node.try_get_context("region") or "eu-west-1"
    ),
    description="Tourist Prediction MLOps Platform — ECS Fargate + ALB + S3 + CloudWatch"
)

app.synth()
