# Offline Multi-Object NeRF

Our multi-object NeRF system is decoupled from the object SLAM system and used as a dynamic link library. An offline version that can be run independently is provided here, which is only applicable to the synthetic sequence. It is simpler and lighter than the online version combined with object SLAM.

## Building

Please fulfill the prerequisites in the main README file first. Then use CMake to build tiny-cuda-nn and Multi-Ojbect-NeRF.

```
sh build.sh
```

## Run

The offline version is only available to the synthetic sequence, please download first. Then run:


```bash
cd dependencies/Multi-Object-NeRF

# Specify which GPU to use (one or two are recommended)
export CUDA_VISIBLE_DEVICES=0

# Since the visualization is implemented using OpenGL, set the environment variable to make it run on the GPU.
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./build/OfflineNeRF ./Core/configs/base.json [path_to_sequence] [Use_GTdepth(0 or 1)]
```
