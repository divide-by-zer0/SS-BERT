python scripts/score_splice_sites_tsv.py \
 --in_tsv test_data/Dmel/test_data100.tsv \
 --donor_ckpt models/Dmel/donor_model/100bp/dmel-donor-100bp.pt \
 --acceptor_ckpt models/Dmel/acceptor_model/100bp/dmel-acceptor-100bp.pt \
 --out_tsv evaluate/Dmel/100bp.scored.tsv \
 --batch_size 512 \
 --num_workers 0
python scripts/min_intron_sum.py \
 --in_tsv evaluate/Dmel/100bp.scored.tsv \
 --out_tsv evaluate/Dmel/snpr_100bp.tsv
python scripts/score_splice_sites_tsv.py \
 --in_tsv test_data/Athal/test_data40.tsv \
 --donor_ckpt models/Athal/donor_model/40bp/athal-donor-40bp.pt \
 --acceptor_ckpt models/Athal/acceptor_model/40bp/athal-acceptor-40bp.pt \
 --out_tsv evaluate/Athal/40bp.scored.tsv \
 --batch_size 512 \
 --num_workers 0
python scripts/min_intron_sum.py \
 --in_tsv evaluate/Athal/40bp.scored.tsv \
 --out_tsv evaluate/Athal/snpr_40bp.tsv
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