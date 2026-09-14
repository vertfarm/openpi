

bash /data/keti/snu/workspace/worktrees/pi05-droid-jointpos-velocity/scripts/start_pi05_droid_jointpos_velocity_ours_t2cfg.sh 3

을 통해 /data/keti/snu/workspace/ckpt/team1/hard5_ours_t2cfg_s20_pytorch 를 loading하여 base PI0.5를 대신해서 평가하고자 합니다. 현재는 

bash /data/keti/snu/workspace/worktrees/pi05-droid-jointpos-velocity/scripts/start_pi05_droid_jointpos_velocity_ours_t2cfg.sh  
에서 basePI0.5를 통해 평가하고 있습니다.

해당 .sh 파일에서는 /data/keti/snu/workspace/worktrees/pi05-droid-jointpos-velocity/scripts/serve_pi05_droid_jointpos_velocity_safetensors.py를 실행하게 됩니다.

data/keti/snu/workspace/worktrees/pi05-droid-jointpos-velocity/scripts/start_pi05_droid_jointpos_velocity_ours_t2cfg.sh

와

/data/keti/snu/workspace/worktrees/pi05-droid-jointpos-velocity/scripts/serve_pi05_droid_jointpos_velocity_safetensors.py 만을 수정하여 위의 saftetensors를 정상 loading할 수 있도록 해주세요.

시뮬레이션에서 사용한 로딩 방법은 /data/keti/snu/workspace/openpi_bhl/docs/260914_checkpoint_loading.md를 확인하면 됩니다.

/data/keti/snu/workspace/ckpt/team1/hard5_base_ppo_s80_pytorch
/data/keti/snu/workspace/ckpt/team1/hard5_opsd_base_s80_pytorch

등도 순차적으로 로딩 후에 평가할 것입니다.