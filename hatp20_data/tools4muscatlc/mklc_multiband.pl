#!/usr/bin/perl
## mklc_multiband.pl
## written by A. Fukui, July 3, 2017

($list) = @ARGV;
unless(@ARGV==1){
  print "\n";
  print "Usage: perl mklc_multiband.pl [input list]\n";
  print "\n";
  print "  Note 1. The input list must be provided in the following format.\n";
#  print "    column1: band id (must be consecutive starting with 0)\n";
  print "    column1: light curve (magnitude scale)\n"; 
  print "    column2: exposure time (sec)\n";
  print "    column3: aperture radius (pix)\n";
  print "    Ex.)\n";
  print "      lcm_msct_g_hatp12_170124_t1_c23_r28-36.bjd.dat 60 30\n";
  print "      lcm_msct_r_hatp12_170124_t1_c23_r28-36.bjd.dat 20 32\n";
  print "      lcm_msct_z_hatp12_170124_t1_c23_r28-36.bjd.dat 50 32\n";
  print "\n";
  print "  Note 2. Comparison stars must be common among all light curves in the list.\n";
#  print "  Note 3. The second argument must be the band id for the target star. A raw light curve of the target star in this band will be produced along with the raw light curves of ensemble of the comparison stars in all bands.\n";
  print "\n";
  exit;
}

$deadtime = 4; # sec

$id=0;
open(IN, $list) or die;
while($line=<IN>){
  next if $line=~/#/;
  ($lc,$exp,$rad) = split(" ",$line,4); 
  $lc[$id] = $lc;
  $exp[$id] = $exp;
  $rad[$id] = sprintf("%.1f",$rad);
  ($gar,$lc_tmp) = split("lcm_",$lc);
  @words = split("_",$lc_tmp);
  $inst[$id] = $words[0];
  $band[$id] = $words[1];
  $target[$id] = $words[2];
  $date[$id] = $words[3];
  $tID[$id] = $words[4];
  $cID[$id] = $words[5];
  $id++;
}
close IN;
$nid=$id;


## loop for tid
for($tid=0; $tid<$nid; $tid++){

  ## load light curve of the target band
  $j=0;
  open(IN, $lc[$tid]) or die;
  while($line=<IN>){
    @words = split(" ",$line);
    if($line=~/JD/){
      $tcol=$tcecol=$ccol="";
      for($i=0; $i<@words; $i++){
        $tcol=$i-1 if $words[$i] eq "mag_t(r=$rad[$tid])";
        $tcecol=$i-1 if $words[$i] eq "mag_t-c_err(r=$rad[$tid])";
        $ccol=$i-1 if $words[$i] eq "mag_c(r=$rad[$tid])";
      } 
      unless($tcol || $tcecol || $ccol){
        print "No data entry in band_id=$tid for rad=$rad[$tid]\n";
        exit;
      }
      $bjdflag=0;
      $bjdflag=1 if $line=~/BJD/;
      next;
    }
    next if $line=~/#/;
    ($tt[$j],$ta[$j],$ts[$j],$tdx[$j],$tdy[$j],$tfwhm[$j],$tpeak[$j],$tframe[$j])
    =($words[0],$words[1],$words[2],$words[3],$words[4],$words[5],$words[6],$words[7]);
    $tmag[$j] = $words[$tcol];
  #  $cmag[$tid][$j] = $words[$ccol];
    $cmag_intp[$tid][$j] = $words[$ccol];
    $tcmage[$j] = $words[$tcecol];
    $j++;
  }
  close IN;
  $nt=$j;
  

  ## load light curves of the other bands
  for($id=0; $id<$nid; $id++){

    next if $id == $tid;

    $nc[$id]=0;
    open(IN, $lc[$id]) or die;
    while($line=<IN>){
      @words = split(" ",$line);
      if($line=~/JD/){
        $ccol="";
        for($i=0; $i<@words; $i++){
          $ccol=$i-1 if $words[$i] eq "mag_c(r=$rad[$id])";
        } 
        unless($ccol){
          print "No data entry in band_id=$id for rad=$rad[$id]\n";
          exit;
        }
        next;
      }
      next if $line=~/#/;
      $ct[$id][$nc[$id]] = $words[0];
      $cmag[$id][$nc[$id]] = $words[$ccol];
      $nc[$id]++;
    }
    close IN;

  }


  for($id=0; $id<$nid; $id++){

    next if $id == $tid;

    $dt = ($exp[$tid]+$deadtime)/86400 if $exp[$tid] >= $exp[$id];
    $dt = ($exp[$id]+$deadtime)/86400 if $exp[$tid] < $exp[$id];

    for($i=0; $i<$nt; $i++){

      # search for cmag data within the time window of 2 x dt
      $k=0;
      @datatmp=();
      for($j=0; $j<$nc[$id]; $j++){

        # no interpolation is applied if abs(tt-ct)/dt < 0.1 && exp[tid]<=exp[id]
        if( abs($tt[$i]-$ct[$id][$j])/$dt < 0.1 && $exp[$tid]<=$exp[$id]){
          $cmag_intp[$id][$i] = $cmag[$id][$j];
          $k=-1;
          last;
        }

        if($ct[$id][$j] >= $tt[$i]-$dt && $ct[$id][$j] < $tt[$i]+$dt){
          $ct_tmp[$k] = $ct[$id][$j];
          $cmag_tmp[$k] = $cmag[$id][$j]; 
          $k++;
        }
      }
      next if $k==-1;

      # search again, with a x2 wider time window, if k=1 & 2/3*exp[tid] < exp[id]
      $rep=0;
      if($k==1 && 2/3*$exp[$tid] < $exp[$id]){
        $k=0;
        @datatmp=();
        for($j=0; $j<$nc[$id]; $j++){
          if($ct[$id][$j] >= $tt[$i]-2*$dt && $ct[$id][$j] < $tt[$i]+2*$dt){
            $ct_tmp[$k] = $ct[$id][$j];
            $cmag_tmp[$k] = $cmag[$id][$j];
            $k++;
            $rep=1;
          }
        }
      }

      if($k<2){
        $cmag_intp[$id][$i] = 0; 
        next;
      }
      $npt=$k;

      # inter(extra)polation by linear fit 
      $sumx=$sumy=$sumx2=$sumxy=0;
      for($k=0; $k<$npt; $k++){
        $DT = $ct_tmp[$k]-$tt[$i];
        $sumx+= $DT;
        $sumx2+= $DT**2;
        $sumy+= $cmag_tmp[$k];
        $sumxy+= $DT*$cmag_tmp[$k];
      }
      $D = $npt*$sumx2 - $sumx**2;
      $A = ($sumx2*$sumy - $sumx*$sumxy)/$D;
      $B = ($npt*$sumxy - $sumx*$sumy)/$D;

      $cmag_intp[$id][$i] = $A;

#    print "id=$id i=$i npt=$npt rep=$rep $A $B $cmag_intp[$id][$i]\n";

    }

  }

  $output = "lcm_$inst[$tid]_multi_$band[$tid]_$target[$tid]_$date[$tid]_$tID[$tid]_$cID[$tid].dat";
  print "output=$output\n";
  open(OUT, ">$output") or die;

  if($bjdflag){
    print OUT "# BJD(TDB)-2450000 airmass sky(mag) dx(pix) dy(pix) fwhm(pix) peak(mag) frame mag_t(band=$tid,r=$rad[$tid]) mag_t-c_err(band=$tid,r=$rad[$tid]) ";
  }else{
    print OUT "# GJD-2450000 airmass sky(mag) dx(pix) dy(pix) fwhm(pix) peak(mag) frame mag_t(band=$tid,r=$rad[$tid]) mag_t-c_err(band=$tid,r=$rad[$tid])";
  }

  for($id=0; $id<$nid; $id++){
    print OUT " mag_c(band=$id,r=$rad[$id])";
  } 
  print OUT "\n";


  for($i=0; $i<$nt; $i++){

    $skip=0;
    for($id=0; $id<$nid; $id++){
      $skip=1 if $cmag_intp[$id][$i]==0;
    }
    next if $skip==1;
    
    print OUT "$tt[$i] $ta[$i] $ts[$i] $tdx[$i] $tdy[$i] $tfwhm[$i] $tpeak[$i] $tframe[$i] $tmag[$i] $tcmage[$i] ";

    for($id=0; $id<$nid; $id++){
      print OUT "$cmag_intp[$id][$i] ";
    } 
    print OUT "\n";

  }
  close OUT;

} ## end of loop for tid
