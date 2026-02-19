#40bp Athal
python scripts/min_intron_sum.py \
 --in_tsv evaluate/Athal/40bp.scored.tsv \
 --out_tsv evaluate/Athal/40bp_min_prod.tsv \
 --combine product
zsh evaluate/evaluate_scores.sh raw_files/Athal/transcript_class.txt evaluate/Athal/40bp_min_prod.tsv
 #40bp Mmus
python scripts/min_intron_sum.py \
 --in_tsv evaluate/Mmus/40bp.scored.tsv \
 --out_tsv evaluate/Mmus/40bp_min_prod.tsv \
 --combine product
zsh evaluate/evaluate_scores.sh raw_files/Mmus/transcript_class.txt evaluate/Mmus/40bp_min_prod.tsv
