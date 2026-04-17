# Setting up lane graph files
Current version of `scorer` requires pre-extracted lane graph information.  
In order to run `scripts/extract_roadgraph.py` you should separately install `waymo-open-dataset` codes.  
First, clone the `waymo-open-dataset` repository into the project directory.  
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
Then you can set up lane graph files by running
```
cd scripts
python extract_roadgraph.py --split <training or validation> --output_dir <ex. /zfsauton/scratch/mineuih/waymax_rs/lane_graphs>
```

# Run goal reaching experiment
```
cd waymax_rs
python scripts/main_goal_reaching.py --checkpoint_path /path_to_ckpt/epoch_0100/ --split <training or validation> --exp_dir /path_to_save_results/exp/ --lane_graph_dir /output_path_of_extract_roadgraph.py/ --num_worlds <number of worlds run in parallel> --num_scenarios <total number of scenarios you want to test> --use_es