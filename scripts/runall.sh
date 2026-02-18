#100bp Dmel
python scripts/ssbert.py \
  --task acceptor \
  --pos train_data/Dmel/100bp_window/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.pwm.err \
  --neg train_data/Dmel/100bp_window/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.neg.pwm.err \
  --out_dir models/Dmel/acceptor_model/100bp \
  --k 3 --max_len 100 \
  --d_model 256 --n_heads 8 --n_layers 6 --dropout 0.1 \
  --batch_size 256 --epochs 20 \
  --lr 1e-4 --weight_decay 0.01 \
  --val_frac 0.1 \
  --num_workers 0 \
  --ckpt_name dmel-acceptor-100bp.pt
python scripts/ssbert.py \
  --task donor \
  --pos train_data/Dmel/100bp_window/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.pwm.err \
  --neg train_data/Dmel/100bp_window/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.neg.pwm.err \
  --out_dir models/Dmel/donor_model/100bp \
  --k 3 --max_len 100 \
  --d_model 256 --n_heads 8 --n_layers 6 --dropout 0.1 \
  --batch_size 256 --epochs 20 \
  --lr 1e-4 --weight_decay 0.01 \
  --val_frac 0.1 \
  --num_workers 0 \
  --ckpt_name dmel-donor-100bp.pt
python scripts/score_splice_sites_tsv.py \
 --in_tsv test_data/Dmel/test_data100.tsv \
 --donor_ckpt models/Dmel/donor_model/100bp/dmel-donor-100bp.pt \
 --acceptor_ckpt models/Dmel/acceptor_model/100bp/dmel-acceptor-100bp.pt \
 --out_tsv evaluate/Dmel/100bp.scored.tsv \
 --batch_size 512 \
 --num_workers 0
python scripts/min_intron_sum.py \
 --in_tsv evaluate/Dmel/100bp.scored.tsv \
 --out_tsv evaluate/Dmel/100bp_min.tsv

#40bp Athal (using 100bp training data, clipped to 40bp via --max_len)
python scripts/ssbert.py \
  --task acceptor \
  --pos train_data/Athal/100bp_window/GCF_000001735.4_TAIR10.1_genomic.clean.fna.pwm.err \
  --neg train_data/Athal/100bp_window/GCF_000001735.4_TAIR10.1_genomic.clean.fna.neg.pwm.err \
  --out_dir models/Athal/acceptor_model/40bp \
  --k 3 --max_len 40 \
  --d_model 256 --n_heads 8 --n_layers 6 --dropout 0.1 \
  --batch_size 256 --epochs 20 \
  --lr 1e-4 --weight_decay 0.01 \
  --val_frac 0.1 \
  --num_workers 0 \
  --ckpt_name athal-acceptor-40bp.pt
python scripts/ssbert.py \
  --task donor \
  --pos train_data/Athal/100bp_window/GCF_000001735.4_TAIR10.1_genomic.clean.fna.pwm.err \
  --neg train_data/Athal/100bp_window/GCF_000001735.4_TAIR10.1_genomic.clean.fna.neg.pwm.err \
  --out_dir models/Athal/donor_model/40bp \
  --k 3 --max_len 40 \
  --d_model 256 --n_heads 8 --n_layers 6 --dropout 0.1 \
  --batch_size 256 --epochs 20 \
  --lr 1e-4 --weight_decay 0.01 \
  --val_frac 0.1 \
  --num_workers 0 \
  --ckpt_name athal-donor-40bp.pt
python scripts/score_splice_sites_tsv.py \
 --in_tsv test_data/Athal/test_data40.tsv \
 --donor_ckpt models/Athal/donor_model/40bp/athal-donor-40bp.pt \
 --acceptor_ckpt models/Athal/acceptor_model/40bp/athal-acceptor-40bp.pt \
 --out_tsv evaluate/Athal/40bp.scored.tsv \
 --batch_size 512 \
 --num_workers 0
python scripts/min_intron_sum.py \
 --in_tsv evaluate/Athal/40bp.scored.tsv \
 --out_tsv evaluate/Athal/40bp_min.tsv

#40bp Mmus (using 100bp training data, clipped to 40bp via --max_len)
python scripts/ssbert.py \
  --task acceptor \
  --pos train_data/Mmus/100bp_window/GCF_000001635.27_GRCm39_genomic.fna.pwm.err \
  --neg train_data/Mmus/100bp_window/GCF_000001635.27_GRCm39_genomic.fna.neg.pwm.err \
  --out_dir models/Mmus/acceptor_model/40bp \
  --k 3 --max_len 40 \
  --d_model 256 --n_heads 8 --n_layers 6 --dropout 0.1 \
  --batch_size 256 --epochs 20 \
  --lr 1e-4 --weight_decay 0.01 \
  --val_frac 0.1 \
  --num_workers 0 \
  --ckpt_name mmus-acceptor-40bp.pt
python scripts/ssbert.py \
  --task donor \
  --pos train_data/Mmus/100bp_window/GCF_000001635.27_GRCm39_genomic.fna.pwm.err \
  --neg train_data/Mmus/100bp_window/GCF_000001635.27_GRCm39_genomic.fna.neg.pwm.err \
  --out_dir models/Mmus/donor_model/40bp \
  --k 3 --max_len 40 \
  --d_model 256 --n_heads 8 --n_layers 6 --dropout 0.1 \
  --batch_size 256 --epochs 20 \
  --lr 1e-4 --weight_decay 0.01 \
  --val_frac 0.1 \
  --num_workers 0 \
  --ckpt_name mmus-donor-40bp.pt
python scripts/score_splice_sites_tsv.py \
 --in_tsv test_data/Mmus/test_data40.tsv \
 --donor_ckpt models/Mmus/donor_model/40bp/mmus-donor-40bp.pt \
 --acceptor_ckpt models/Mmus/acceptor_model/40bp/mmus-acceptor-40bp.pt \
 --out_tsv evaluate/Mmus/40bp.scored.tsv \
 --batch_size 512 \
 --num_workers 0
python scripts/min_intron_sum.py \
 --in_tsv evaluate/Mmus/40bp.scored.tsv \
 --out_tsv evaluate/Mmus/40bp_min.tsv