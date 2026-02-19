#100bp Dmel
# python scripts/ssbert.py \
#   --task acceptor \
#   --pos train_data/Dmel/100bp_window/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.pwm.err \
#   --neg train_data/Dmel/100bp_window/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.neg.pwm.err \
#   --out_dir models/Dmel/acceptor_model/100bp \
#   --k 3 --max_len 100 \
#   --d_model 256 --n_heads 8 --n_layers 6 --dropout 0.1 \
#   --batch_size 256 --epochs 20 \
#   --lr 1e-4 --weight_decay 0.01 \
#   --val_frac 0.1 \
#   --num_workers 0 \
#   --loss focal \
#   --focal_gamma 3.0 \
#   --ckpt_name dmel-acceptor-100bp-focal.pt
# python scripts/ssbert.py \
#   --task donor \
#   --pos train_data/Dmel/100bp_window/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.pwm.err \
#   --neg train_data/Dmel/100bp_window/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.neg.pwm.err \
#   --out_dir models/Dmel/donor_model/100bp \
#   --k 3 --max_len 100 \
#   --d_model 256 --n_heads 8 --n_layers 6 --dropout 0.1 \
#   --batch_size 256 --epochs 20 \
#   --lr 1e-4 --weight_decay 0.01 \
#   --val_frac 0.1 \
#   --num_workers 0 \
#   --loss focal \
#   --focal_gamma 3.0 \
#   --ckpt_name dmel-donor-100bp-focal.pt
# python scripts/score_splice_sites_tsv.py \
#  --in_tsv test_data/Dmel/test_data100.tsv \
#  --donor_ckpt models/Dmel/donor_model/100bp/dmel-donor-100bp-focal.pt \
#  --acceptor_ckpt models/Dmel/acceptor_model/100bp/dmel-acceptor-100bp-focal.pt \
#  --out_tsv evaluate/Dmel/100bp.scored-focal.tsv \
#  --batch_size 512 \
#  --num_workers 0
# python scripts/min_intron_sum.py \
#  --in_tsv evaluate/Dmel/100bp.scored-focal.tsv \
#  --out_tsv evaluate/Dmel/100bp_min-focal-prod.tsv \
#  --combine product
# zsh evaluate/evaluate_scores.sh raw_files/Dmel/transcript_class.txt evaluate/Dmel/100bp_min-focal-prod.tsv

#100bp Athal
python scripts/ssbert.py \
  --task acceptor \
  --pos train_data/Athal/100bp_window/GCF_000001735.4_TAIR10.1_genomic.clean.fna.pwm.err \
  --neg train_data/Athal/100bp_window/GCF_000001735.4_TAIR10.1_genomic.clean.fna.neg.pwm.err \
  --out_dir models/Athal/acceptor_model/100bp \
  --k 3 --max_len 100 \
  --d_model 256 --n_heads 8 --n_layers 6 --dropout 0.1 \
  --batch_size 256 --epochs 20 \
  --lr 1e-4 --weight_decay 0.01 \
  --val_frac 0.1 \
  --num_workers 0 \
  --ckpt_name athal-acceptor-100bp.pt
python scripts/ssbert.py \
  --task donor \
  --pos train_data/Athal/100bp_window/GCF_000001735.4_TAIR10.1_genomic.clean.fna.pwm.err \
  --neg train_data/Athal/100bp_window/GCF_000001735.4_TAIR10.1_genomic.clean.fna.neg.pwm.err \
  --out_dir models/Athal/donor_model/100bp \
  --k 3 --max_len 100 \
  --d_model 256 --n_heads 8 --n_layers 6 --dropout 0.1 \
  --batch_size 256 --epochs 20 \
  --lr 1e-4 --weight_decay 0.01 \
  --val_frac 0.1 \
  --num_workers 0 \
  --ckpt_name athal-donor-100bp.pt
python scripts/score_splice_sites_tsv.py \
 --in_tsv test_data/Athal/test_data40.tsv \
 --donor_ckpt models/Athal/donor_model/100bp/athal-donor-100bp.pt \
 --acceptor_ckpt models/Athal/acceptor_model/100bp/athal-acceptor-100bp.pt \
 --out_tsv evaluate/Athal/40bp.scored.tsv \
 --batch_size 512 \
 --num_workers 0
python scripts/min_intron_sum.py \
 --in_tsv evaluate/Athal/100bp.scored.tsv \
 --out_tsv evaluate/Athal/100bp_min-focal-prod.tsv \
 --combine product
zsh evaluate/evaluate_scores.sh raw_files/Athal/transcript_class.txt evaluate/Athal/100bp_min-focal-prod.tsv
#100bp Mmus 
python scripts/ssbert.py \
  --task acceptor \
  --pos train_data/Mmus/100bp_window/GCF_000001635.27_GRCm39_genomic.fna.pwm.err \
  --neg train_data/Mmus/100bp_window/GCF_000001635.27_GRCm39_genomic.fna.neg.pwm.err \
  --out_dir models/Mmus/acceptor_model/100bp \
  --k 3 --max_len 100 \
  --d_model 256 --n_heads 8 --n_layers 6 --dropout 0.1 \
  --batch_size 256 --epochs 30 \
  --lr 1e-4 --weight_decay 0.01 \
  --val_frac 0.1 \
  --num_workers 0 \
  --ckpt_name mmus-acceptor-100bp.pt
python scripts/ssbert.py \
  --task donor \
  --pos train_data/Mmus/100bp_window/GCF_000001635.27_GRCm39_genomic.fna.pwm.err \
  --neg train_data/Mmus/100bp_window/GCF_000001635.27_GRCm39_genomic.fna.neg.pwm.err \
  --out_dir models/Mmus/donor_model/100bp \
  --k 3 --max_len 100 \
  --d_model 256 --n_heads 8 --n_layers 6 --dropout 0.1 \
  --batch_size 256 --epochs 30 \
  --lr 1e-4 --weight_decay 0.01 \
  --val_frac 0.1 \
  --num_workers 0 \
  --ckpt_name mmus-donor-100bp.pt
python scripts/score_splice_sites_tsv.py \
 --in_tsv test_data/Mmus/test_data40.tsv \
 --donor_ckpt models/Mmus/donor_model/100bp/mmus-donor-100bp.pt \
 --acceptor_ckpt models/Mmus/acceptor_model/100bp/mmus-acceptor-100bp.pt \
 --out_tsv evaluate/Mmus/100bp.scored.tsv \
 --batch_size 512 \
 --num_workers 0
python scripts/min_intron_sum.py \
 --in_tsv evaluate/Mmus/100bp.scored.tsv \
 --out_tsv evaluate/Mmus/100bp_min-focal-prod.tsv \
 --combine product
zsh evaluate/evaluate_scores.sh raw_files/Mmus/transcript_class.txt evaluate/Mmus/100bp_min-focal-prod.tsv