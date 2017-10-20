#!/usr/bin/perl

($cut_list, $lc_list) = @ARGV;
unless(@ARGV==2){
  print "\n";
  print "Usage: perl remove_outliers.pl [cut list] [light curve]\n";
  print "\n";
  print "  Note1: [cut list] must be  a list of frame names of outlires.\n";
  print "  Note2: [light curve] must be a light curve with .dat extension. A new light curve with outliers excluded will be created with .cut.dat extension.\n";
  print "\n";
  exit;
}

#unless(@ARGV >= 1){
#   print "Usage:\n";
#  print "  -cut_list [list of framenames of outliers\n";
#  print "  -lc_list [list of files (with .dat extension) from which outliers are excluded]\n";
#  exit;
#}

#for($i=0; $i<@ARGV; $i++){
#  $cut_list = $ARGV[$i+1] if $ARGV[$i] eq "-cut_list";
#  $lc_list = $ARGV[$i+1] if $ARGV[$i] eq "-lc_list";
#}

$com="";
open(IN, $cut_list) or die;
while($line=<IN>){
  next if $line=~/#/;
  chomp $line;
  $com .= "-e \'\/$line\/d\' ";
}
close IN;


open(IN, $lc_list) or die;
while($line=<IN>){
  next if $line=~/#/;
  chomp $line;
  $file = $line;
  
  ($head,$rem) = split(".dat",$file);
  if(-f $file){

    $newfile = "$head.cut.dat";
    print "cp $file $newfile\n";
    system "cp $file $newfile";
  
    print "sed $com $file > $newfile\n";
    system "sed $com $file > $newfile";
  }
}
close IN;
