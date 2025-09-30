import argparse
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def validate_model(model_path: str = "models/sentiment") -> bool:
    """Run validation tests before deployment"""
    print("\n🧪 Running model validation...")

    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        return False

    # Run evaluation script
    cmd = ["python", "scripts/evaluate.py", "--model-path", model_path, "--threshold", "0.85"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Model validation passed")
            return True
        else:
            print("❌ Model validation failed")
            return False
    except Exception as e:
        print(f"❌ Error running validation: {e}")
        return False


def deploy_to_huggingface(space_name: str, token: str):
    """Deploy application to Hugging Face Spaces"""
    api = HfApi()

    # crete or update space
    try:
        repo_url = create_repo(
            repo_id=space_name, repo_type="space", space_sdk="gradio", token=token, exist_ok=True
        )
        print(f"Space created/updated: {repo_url}")
    except Exception as e:
        print(f"Error creating space: {e}")
        return False

    # prepare files for deployment
    space_files = {"app.py": "app/gradio_app.py", "requirements.txt": "requirements/base.txt"}

    # copy source files
    for space_file, local_file in space_files.items():
        api.upload_file(
            path_or_fileobj=local_file,
            path_in_repo=space_file,
            repo_id=space_name,
            repo_type="space",
            token=token,
        )

    # upload source directory
    api.upload_folder(
        folder_path="src", path_in_repo="src", repo_id=space_name, repo_type="space", token=token
    )

    print("✅ Deployment Successful")
    print(f"🚀 Access your app at: https://huggingface.co/spaces/{space_name}")
    return True


def deploy_docker(image_name: str, registry: str):
    """Deploy Docker container"""
    # build image
    build_cmd = f"docker build -f docker/Dockerfile -t {image_name} ."
    subprocess.run(build_cmd, shell=True, check=True)

    # tag for registry
    tag_cmd = f"docker tag {image_name} {registry}/{image_name}"
    subprocess.run(tag_cmd, shell=True, check=True)

    # push to registry
    push_cmd = f"docker push {registry}/{image_name}"
    subprocess.run(push_cmd, shell=True, check=True)

    print(f"Docker image pushed to {registry}/{image_name}")


def main():
    parser = argparse.ArgumentParser(description="Deploy sentiment analysis service")
    parser.add_argument("--space-name", type=str, help="Hugging Face space name")
    parser.add_argument("--token", type=str, help="Hugging Face token")
    parser.add_argument("--docker", action="store_true", help="Deploy as Docker container")
    parser.add_argument("--registry", type=str, help="Docker registry")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["dev", "staging", "production"],
        default="dev",
        help="Deployment stage",
    )
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip model validation before deployment"
    )
    parser.add_argument(
        "--model-path", type=str, default="models/sentiment", help="Path to model directory"
    )

    args = parser.parse_args()

    # Run validation unless skipped
    if not args.skip_validation:
        if not validate_model(args.model_path):
            print("❌ Deployment aborted due to validation failure")
            return 1

    # Deploy to HuggingFace Spaces
    if args.space_name and args.token:
        deploy_to_huggingface(args.space_name, args.token)

    # Deploy as Docker container
    if args.docker:
        image_name = f"sentiment-mlops:{args.stage}"
        deploy_docker(image_name, args.registry or "ghcr.io")

    return 0


if __name__ == "__main__":
    main()
