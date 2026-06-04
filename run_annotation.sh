
source ~/.bashrc
cd ~/waymax_rs
conda activate waymax_rs

python -m language.manual_annotation.cached_annotation.annotation --cpu --num_workers 64 ; \
python -m language.manual_annotation.cached_annotation.annotation --cpu --num_workers 64 --start_timestep 10 ; \
python -m language.manual_annotation.cached_annotation.annotation --cpu --num_workers 64 --start_timestep 20 ; \
python -m language.manual_annotation.cached_annotation.annotation --cpu --num_workers 64 --start_timestep 30 ; \
python -m language.manual_annotation.cached_annotation.annotation --cpu --num_workers 64 --start_timestep 40 ; \
python -m language.manual_annotation.cached_annotation.annotation --cpu --num_workers 64 --start_timestep 50 ; \
python -m language.manual_annotation.cached_annotation.annotation --cpu --num_workers 64 --start_timestep 60 ; \
python -m language.manual_annotation.cached_annotation.annotation --cpu --num_workers 64 --start_timestep 70 ; \
python -m language.manual_annotation.cached_annotation.annotation --cpu --num_workers 64 --start_timestep 80