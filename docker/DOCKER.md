# 🐋 Docker image for contributors

This Docker image is designed for users and contributors who want to run experiments with `torch-uncertainty` on remote virtual machines with GPU support. It is particularly useful for those who do not have access to a local GPU and need a pre-configured environment for development and experimentation.

---

## How to Use The Docker Image

### Step 1: Fork the Repository

Before proceeding, ensure you have forked the `torch-uncertainty` repository to your own GitHub account. You can do this by visiting the [torch-uncertainty GitHub repository](https://github.com/torch-uncertainty/torch-uncertainty) and clicking the **Fork** button in the top-right corner.

Once forked, clone your forked repository to your local machine:

```bash
git clone git@github.com:<your-username>/torch-uncertainty.git
cd torch-uncertainty
```

> ### ⚠️ IMPORTANT NOTE: Keep Your Fork Synced
>
> **To ensure that you are working with the latest stable version and bug fixes, you must manually sync your fork with the upstream repository before building the Docker image. Failure to sync your fork may result in outdated dependencies or missing bug fixes in the Docker image.**

### Step 2: Build the Docker image locally

Build the modified image locally and push it to a Docker registry. If using a cpu, type:

```bash
docker build --build-arg UV_EXTRA=cpu -t my-torch-uncertainty-docker:version . 
```

Otherwise:

```bash
docker build --build-arg UV_EXTRA=gpu -t my-torch-uncertainty-docker:version . 
```

Then:

```bash
docker push my-dockerhub-user/my-torch-uncertainty-image:version
```

### Step 3: Set environment variables on your VM

Connect to you VM and set the following environment variables:

```bash
export VM_SSH_PUBLIC_KEY="$(cat ~/.ssh/id_rsa.pub)"
export GITHUB_SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)"
export GITHUB_USER="your-github-username"
export GIT_USER_EMAIL="your-email@example.com"
export GIT_USER_NAME="Your Name"
export USE_COMPACT_SHELL_PROMPT=true
```

Here is a brief explanation of the environment variables used in the Docker setup:

- **`VM_SSH_PUBLIC_KEY`**: The public SSH key used to authenticate with the container via SSH.
- **`GITHUB_SSH_PRIVATE_KEY`**: The private SSH key used to authenticate with GitHub for cloning and pushing repositories.
- **`GITHUB_USER`**: The GitHub username used to clone the repository during the first-time setup.
- **`GIT_USER_EMAIL`**: The email address associated with the Git configuration for commits.
- **`GIT_USER_NAME`**: The name associated with the Git configuration for commits.
- **`USE_COMPACT_SHELL_PROMPT`** (optional): Enables a compact and colorized shell prompt inside the container if set to `"true"`.

### Step 4: Run the Docker container

First, authenticate with your Docker registry if you use a private registry.
Then run the following command to run the Docker image from your docker registriy

```bash
docker run --rm -it --gpus all -p 8888:8888 -p 22:22 \
    -e VM_SSH_PUBLIC_KEY \
    -e GITHUB_SSH_PRIVATE_KEY \
    -e GITHUB_USER \
    -e GIT_USER_EMAIL \
    -e GIT_USER_NAME \
    -e USE_COMPACT_SHELL_PROMPT \
    docker.io/my-dockerhub-user/my-torch-uncertainty-image:version
```

### Step 5: Connect to your container

Once the container is up and running, you can connect to it via SSH:  

```bash
ssh -i /path/to/private_key root@<VM_HOST> -p <VM_PORT>
```

Replace `<VM_HOST>` and `<VM_PORT>` with the host and port of your VM,  
and `/path/to/private_key` with the private key that corresponds to `VM_SSH_PUBLIC_KEY`.

The container exposes port `8888` in case you want to run Jupyter Notebooks or TensorBoard.

**Note:** The `/workspace` directory is mounted from your local machine or cloud storage,  
so changes persist across container restarts.  
If using a cloud provider, ensure your network volume is correctly attached to avoid losing data.

## Remote Development

This Docker setup also allows for remote development on the VM, since GitHub SSH access is set up and the whole repo is cloned to the VM from your GitHub fork.
For example, you can seamlessly connect your VS Code editor to your remote VM and run experiments, as if on your local machine but with the GPU acceleration of your VM.
See [VS Code Remote Development](https://code.visualstudio.com/docs/remote/remote-overview) for further details.

## Streamline setup with your Cloud provider of choice

Many cloud providers offer "templates" where you can specify a Docker image to use as a base. This means you can:

1. Specify the Docker image from your Docker registry as the base image.
2. Preconfigure the necessary environment variables in the template.
3. Reuse the template any time you need to spin up a virtual machine for experiments.

The cloud provider will handle setting the environment variables, pulling the Docker image, and spinning up the container for you. This approach simplifies the process and ensures consistency across experiments.
