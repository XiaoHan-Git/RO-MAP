cd Core
cd third_party
cd tiny-cuda-nn
git submodule update --init --recursive

cd ../../
echo "Configuring and building Multi-Object-NeRF ..."
cmake . -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -B build
cmake --build build --config RelWithDebInfo --target all -- -j


cd ../
echo "Configuring and building OfflineNeRF ..."
cmake . -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -B build
cmake --build build --config RelWithDebInfo --target all -- -j





