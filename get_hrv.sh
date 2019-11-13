#! /bin/sh

# get_hrv		Joe Mietus		Apr 1 2011

# get_hrv		Joe Mietus		Nov 9 2010


USAGE="$0 [options] -R rrfile | record annotator [start [end]]
	Get HRV statistics :
	  REC : NN/RR AVNN SDNN SDANN SDNNINDX RMSSD PNN : TOTPWR ULF VLF LF HF LF/HF
	options :
	  [-R rrfile] : RR interval file : time (sec), interval
	  [-f \"filt hwin\"] : filter outliers
	  [-p \"nndiff ...\"] : nn difference for pnn (default: 50 msec)
	  [-P \"lo1 hi1 lo2 hi2 lo3 hi3 lo4 hi4\"] : power bands
		(default : 0 0.0033 0.0033 0.04 0.04 0.15 0.15 0.4)
	  [-s] : short term stats of
	         REC : NN/RR AVNN SDNN RMSSD PNN : TOTPWR VLF LF HF LF/HF
	  [-I c|h|m] : input time format: hh::mm:ss, hours, minutes (default: seconds)
          [-m] : RR intervals in msec
	  [-M] : output statistics in msec rather than sec
	  [-L] : output statistics on one line 
	  [-S] : plot HRV results on screen

	plotting options :
	  [-F \"filt hwin\"] : filter outliers, plot filtered data
	  [-y \"ymin ymax\"] : time series y-axis limits (\"- -\" for self-scaling)
	  [-X maxfreq] : fft maximum frequency (default : 0.4 Hz)
	  [-Y fftmax] : fft maximum (\"-\" for self-scaling)
	  [-o] : output plot in postscript

"
# export executable path
export PATH=$PATH:$PWD
while getopts R:f:p:P:sI:mtMLSF:y:X:Y:o c
do
    case $c in
	R) RRFILE=$OPTARG ;;
	f) FILT=$OPTARG ;;
	p) PFLAG="-p $OPTARG" ;;
	P) PWRBANDS=$OPTARG ;;
        s) SFLAG='-s' ;;
        I) TIME=$OPTARG ;;
	m) MFLAG=1 ;;
	M) MSEC='-m' ;;
	L) SLINE=1 ;;
	S) SCREENPLOT=1 ;;
	F) FILT=$OPTARG
	   PLTFILT=1 ;;
	y) Y0LIMS=$OPTARG
	   if test `echo $Y0LIMS | wc -w` -ne 2
	   then
               echo "$0 : [-y \"ymin ymax\"]"
               exit 1
	   fi ;;
	X) X1MAX=$OPTARG ;;
	Y) Y1MAX=$OPTARG ;;
	o) SCREENPLOT=1
	   PS=-ps ;;
	\?) echo "$USAGE"
            exit 1 ;;
    esac
done

let SHIFT_NUMBER=OPTIND-1
shift $SHIFT_NUMBER
if test "$RRFILE"
then
    if test ! -r "$RRFILE"
    then
	echo "$0 : Can't open $RRFILE"
	exit 1
    fi
else
    if test $# -lt 2
    then
	echo "$USAGE"
	exit 1
    fi
    REC=$1
    ANN=$2
    if test ! "`wfdbwhich $REC.$ANN`"
    then
	echo "$0 : can't read annotator $ANN for record $REC"
	exit 1
    fi
    shift 2
fi

START=$1
END=$2
if test ! "$START" -o "$START" = "-"
then
    START=00:00:00
fi
echo "START:"$START
STARTSEC=`seconds $START`
echo "STARTSEC:"$STARTSEC
if test "$STARTSEC" -eq -1
then
    echo "$0 : bad start time : $START"
    exit 1
fi
START=`hours $STARTSEC`
echo "START:"$START
if test "$END"
then 
	echo "END?"
    ENDSEC=`seconds $END`
    if test $STARTSEC -gt $ENDSEC
    then
	echo "$0: start time greater than end time"
	exit 1
    fi
fi

if test ! "$PWRBANDS"
then
    if test -n "$SFLAG"
    then
        LO1=0
        HI1=0.04
        LO2=0.04
        HI2=0.15
        LO3=0.15
        HI3=0.4
    else
        LO1=0
        HI1=0.0033
        LO2=0.0033
        HI2=0.04
        LO3=0.04
        HI3=0.15
        LO4=0.15
        HI4=0.4
    fi
    PWRBANDS="$LO1 $HI1 $LO2 $HI2 $LO3 $HI3 $LO4 $HI4"
    echo "PWRBANDS:"$PWRBANDS
elif test "$PWRBANDS" != 0
then
    if test `echo $PWRBANDS | wc -w | awk '{print $1%2}'` -ne 0
    then
	echo "$0 : [-P \"lo hi [...]\"]"
	exit 1
    fi
fi

#FMAX=`echo $PWRBANDS | tr ' ' '\n' | sort -n | tail -1`
# instead the above statement
FMAX=0.4
(
if test "$RRFILE"
then
    cat $RRFILE
    
else
    rrlist $ANN $REC -f $START ${END:+-t} $END -s
fi
) |
alignTime >foo.rr

cat foo.rr |
statnn $SFLAG $MSEC $PFLAG >foo.nnstat
cat foo.nnstat
cat foo.rr |
filtAnnot |
lomb - |
filtFreqs 0.4 |
pwr $PWRBANDS |
computeLfHfRatio >foo.pwr
cat foo.pwr

