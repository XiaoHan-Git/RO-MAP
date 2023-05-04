cd dependencies/DBoW2
echo "Configuring and building dependencies/DBoW2 ..."
cmake . -B build
cmake --build build --config Release -j

cd ../g2o
echo "Configuring and building dependencies/g2o ..."
cmake . -B build
cmake --build build --config Release -j

cd ../line_lbd
echo "Configuring and building dependencies/line_lbd ..."
cmake . -B build
cmake --build build --config Release -j

cd ../../
echo "Configuring and building RO-MAP ..."
cmake . -B build
cmake --build build --config Release -j

echo "Done ..."
