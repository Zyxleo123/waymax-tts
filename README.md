In order to run `scripts/extract_roadgraph.py` you should separately install `waymo-open-dataset` codes.  
First, clone the `waymo-open-dataset` repository  
```bash
cd waymax_rs
git clone https://github.com/waymo-research/waymo-open-dataset.git
```
Install protobuf comilation tools
```bash
pip install protobuf grpcio-tools
```
Compile protobuf files
```bash
cd waymo-open-dataset
python -m grpc_tools.protoc \
  -I=src \
  --python_out=src \
  src/waymo_open_dataset/*.proto \
  src/waymo_open_dataset/protos/*.proto
```
