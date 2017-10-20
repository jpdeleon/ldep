#!/usr/bin/perl

unless(@ARGV >= 1){
  print "\n";
  print "Usage: perl list_outliers.pl -[parameter] [value]\n";
  print "\n";
  print "## required parameters ##\n";
  print "  -input [input data file]\n";
  print "  -datacol [column# from which outliers will be calculated (start with 1)]\n";
  print "\n";
  print "## optional parameters ##\n";
  print "  -framecol [column# for frame name (start with 1; default=8)]\n";
  print "  -nsmooth [# of adjacent points with which moving average and standard deviation will be calculated (odd number; default=11)]\n";
  print "  -sigma_cut [sigma threshold for outliers (default=4)]\n";
  print "\n";
#  print "  -list [list of files (with .dat extension) from which outliers are excluded]\n";
#  print "  -stdout (a flag to standard output framenames of outliers)\n";
  exit;
}

for($i=0; $i<@ARGV; $i++){
  $input = $ARGV[$i+1] if $ARGV[$i] eq "-input";
  $datacol = $ARGV[$i+1] if $ARGV[$i] eq "-datacol";
  $framecol = $ARGV[$i+1] if $ARGV[$i] eq "-framecol";
  $nsmooth = $ARGV[$i+1] if $ARGV[$i] eq "-nsmooth";
  $sigma_cut = $ARGV[$i+1] if $ARGV[$i] eq "-sigma_cut";
#  $list = $ARGV[$i+1] if $ARGV[$i] eq "-list";
#  $stdout = 1 if $ARGV[$i] eq "-stdout";
}

$nsmooth = 11 unless $nsmooth;
$sigma_cut = 4 unless $sigma_cut;
$framecol = 8 unless $framecol;

unless($datacol){
  print "Please specify datacol.\n";
  exit;
}

print "# outliers picked by smoothing for column $datacol of\n";
print "# $input\n";
print "# with nsmooth=$nsmooth and sigma_cut=$sigma_cut\n";

$i=0;
open(IN, $input) or die;
while($line=<IN>){
  next if $line=~/#/;
  @words = split(" ",$line);
  $data[$i] = $words[$datacol-1];
  $base[$i] = $words[$framecol-1];
  $i++;
}
close IN;
$ndata=$i;

$h=int($nsmooth/2);

$noutlier=1;
while($noutlier){

  @mediandata=();
  for($i=0; $i<$ndata; $i++){
    $k=0;
    @smoothdata=();
    for($j=$i-$h; $j<$i+$h; $j++){
      next if($j<0 || $j>=$ndata);
      $smoothdata[$k] = $data[$j];
      $k++;
    }
    @sort=();
    @sort = sort{$a<=>$b}(@smoothdata);
    $mediandata[$i] = $sort[$k/2];

  }

  $var=0;
  for($i=0; $i<$ndata; $i++){
    $var += ($data[$i]-$mediandata[$i])**2;
  }
  $rms = sqrt($var/$ndata);

  $noutlier=0;
  $j=0;
  for($i=0; $i<$ndata; $i++){
    if(abs($data[$i]-$mediandata[$i]) > $sigma_cut * $rms){
#    if(abs($f[$i]-$smoothf[$i]) > $sigma_cut * $rms){
      print "$base[$i]\n";
      $noutlier++;
    }else{
      $data[$j] = $data[$i];
      $base[$j] = $base[$i];
      $j++;
    }
  }
  $ndata=$j;

}
