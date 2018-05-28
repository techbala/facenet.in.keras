#!/bin/bash

TRAINING_DIRECTORY=lfw.face.aligned.with.321
TEST_DIRECTORY=lfw.face.aligned.with.321.eval
TEST_SAMPLE_COUNTER=4

if [ -d "$TRAINING_DIRECTORY" ]; then
	dirlist=$(find $TRAINING_DIRECTORY/* -maxdepth 1 -type d)
	echo $dirlist
	for dir in $dirlist
        do
        (
            soggetto=$( echo $dir | rev | cut -d '/' -f 1 | rev )
            mkdir -p $TEST_DIRECTORY/$soggetto
            shuf -n $TEST_SAMPLE_COUNTER -e $dir/* | xargs -i mv {} $TEST_DIRECTORY/$soggetto
        )
	done
else 
	echo "Training Directory is not present hence stopped"
fi
