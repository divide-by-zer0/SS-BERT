#!/bin/bash
CLASS_FILE=$1
SCORE_FILE=$2

perl -ane 'BEGIN{
  open(FILE,"'$CLASS_FILE'");
  while($line=<FILE>){
    chomp($line);
    ($tr,$cl)=split(/\s+/,$line);
    $class{$tr}=$cl;
  }
}{
  chomp($F[4]);
  print "$F[0] $F[4] $class{$F[0]}\n" if(defined($class{$F[0]}) && not($class{$F[0]} eq "c") && not($F[-1]==10000));
}' $SCORE_FILE |\
sort -gk2,2 |\
perl -ane 'BEGIN{
  $good=15169;
  $total=38235;
  $ref=32288;
}{
  $good-- if($F[2] eq "=");
  $total--;
  $total=1 if($total==0);
  $sn=int($good/$ref*10000)/100;
  $pr=int($good/$total*10000)/100;
  print "$F[0] $F[1] $F[2] $sn $pr\n";
}' > $SCORE_FILE.eval.txt

