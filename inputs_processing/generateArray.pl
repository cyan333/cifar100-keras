#!/usr/bin/perl
##----------------------------------------------------------------------------------
## Author      : asayal
## Date	       : 06/11/2018
##----------------------------------------------------------------------------------

use strict;
use Getopt::Long;
use File::Basename;
use Cwd;
use Cwd qw(abs_path);
use POSIX qw(ceil); 
use POSIX qw(floor); 
use List::Util qw(min max);
use Data::Dumper qw(Dumper);

## Initializing variables
my $help = 0;
my $input;
my $output_x;
my $output_y;

my @x_array = ();
my @y_array = ();

GetOptions(
    "input=s"      => \$input,
    "output_x=s"   => \$output_x,
    "output_y=s"   => \$output_y,
    "help|h"       => \$help,
);

## MAIN

## Checking inputs
if (! $help) { print "INFO: Checking inputs\n";}
&checkInputs();

## Parsing inputs
print "INFO: Parsing inputs\n";
&parseInputs();

## Generating output reports
print "INFO: Generating outputs\n";
&generateOutputs();

### END OF MAIN ##


sub checkInputs()
{
	if ($help) {&printUsage(); exit 1;}
	chomp($input); $input =~ s/^\s*//g; $input = abs_path($input);
	chomp($output_x); $output_x =~ s/^\s*//g;
	chomp($output_y); $output_y =~ s/^\s*//g;


  	if (($input =~ /^\s*$/) || ($output_x =~ /^\s*$/) || ($output_y =~ /^\s*$/) ) {print "ERROR: Insufficient input arguments\n"; &printUsage(); exit 1;}
  	if (! -e $input) {print "ERROR: Input file specified $input does not exist\n"; exit 1;}

  	my $out_dir = dirname($output_x);
  	if (! -d $out_dir) {`mkdir -p $out_dir`; $output_x = abs_path($output_x); $output_y = abs_path($output_y);}
}

sub parseInputs()
{

	open(IN,$input) || die "ERROR: Cannot open input file $input\n";
	
	my $line_num = 0;

	while (my $line = <IN>)
	{
		chomp($line);
		$line =~ s/^\s*//g;

		my @arr = split(/\s*,\s*/,$line);
		my $size_col = $#arr + 1 ;

		if ($size_col != 3073) {print "ERROR: Incorrect number of columns - $size_col\n"; exit 1;}

		my $channel = 0;
		my $row = 0;
		my $col = 0;

		for(my $i=0;$i<3072;$i++) 
		{
			if (($i == 1024) or ($i == 2048))
			{
				$channel++;
				$row = 0;
				$col = 0;
			}

			my $index = $i - 1024*$channel;


			$row = floor($index/32);
			$col = $index - $row*32;

			$x_array[$line_num][$channel][$row][$col] = $arr[$i];			
		}

		$y_array[$line_num][0] = $arr[$#arr];

		$line_num++;
	}

	close(IN);

	print "INFO: Parsed input files $input\n";

}


sub generateOutputs()
{
	open(OUT_X,">$output_x") || die "ERROR: Cannot open output file $output_x for X vector\n";
	open(OUT_Y,">$output_y") || die "ERROR: Cannot open output file $output_y for Y vector\n";


	print OUT_X Dumper \@x_array ;
	print OUT_Y Dumper \@y_array ;

	close(OUT_X);
	close(OUT_Y);

		## Array size
	my $first_dim = $#x_array + 1;
	my $second_dim = $#{$x_array[0]} + 1;
	my $third_dim = $#{$x_array[0][0]} + 1;
	my $fourth_dim = $#{$x_array[0][0][0]} + 1;

	my $first_y_dim = $#y_array + 1;
	my $second_y_dim = $#{$y_array[0]} + 1;

	print "INFO: X Vector size is $first_dim x $second_dim x $third_dim x $fourth_dim\n";
	print "INFO: Y Vector size is $first_y_dim x $second_y_dim\n";

	print "INFO: Generated output file $output_x for X Vector\n";
	print "INFO: Generated output file $output_y for Y Vector\n";
}

sub printUsage()
{
  print "
  $0   
  -input     <Specify path of input file>        	        (REQUIRED)
  -output_x  <Specify path of output file for X vector>         (REQUIRED)
  -output_y  <Specify path of output file for Y vector>         (REQUIRED)
 ";
}

