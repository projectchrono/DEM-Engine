# Docker Support 
__Run DEM-Engine in a containerized environment__

## Quick start
Make sure to initialize the submodule before using image.

```
git submodule init
git submodule update
```

DEM-Engine provides Docker images for both development and production use:

<li><strong>Production Image</strong>: Optimized runtime environment with pre-built binaries</li>

<li><strong>Development Image</strong>: Full build environment with debugging tools</li>

```
# Build and run the production image
docker-compose -f .devcontainer/docker-compose.yml up deme-production

# Or start a development environment
docker-compose -f .devcontainer/docker-compose.yml run --rm deme-dev
```

## Tips
### Customize output directory & demo
There are two ways to customize the output directory and demo execution:

#### 1. Modify docker-compose.yml
Edit the `docker-compose.yml` file to change default settings:
```
volumes:
  - ../output:/workspace/DEM-Engine/build/output # change this line to your desired output directory
command: ["/workspace/DEM-Engine/build/bin/DEMdemo_Mixer"] # replace with your production command
```
#### 2. Override via Command Line
Use command-line overrides for more flexibility without modifying files:

```
# Run specific demos by overriding the command
docker-compose -f .devcontainer/docker-compose.yml run --rm deme-production \
  /workspace/DEM-Engine/build/bin/DEMdemo_BallDrop

# Run with custom output directory
docker-compose -f .devcontainer/docker-compose.yml run --rm \
  -v $(pwd)/my_output:/workspace/DEM-Engine/build/output \
  deme-production /workspace/DEM-Engine/build/bin/DEMdemo_RotatingDrum

# Run with both custom output directory AND specific demo
docker-compose -f .devcontainer/docker-compose.yml run --rm \
  -v $(pwd)/results/experiment1:/workspace/DEM-Engine/build/output \
  deme-production /workspace/DEM-Engine/build/bin/DEMdemo_ConePenetration
```

### Organizing output files

Based on the current mount configuration, consider using nested directories in your demo scripts for better organization:

```
path out_dir = current_path() / "output" / "DemoOutput_Mixer";
```

## Volume Mounts
The development container uses several volumes for persistence:

<li><strong>deme-build-cache</strong>: Caches build artifacts</li>
<li><strong>deme-ccache</strong>: Compiler cache for faster rebuilds</li>
<li><strong>deme-vscode-extensions</strong>: VSCode extensions</li>




