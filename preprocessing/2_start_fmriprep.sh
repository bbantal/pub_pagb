# Enter subject ID
read -p "Enter subject ID: (XXX format) " sub

# Run fmriprep from within docker
docker run \
--rm \
-v /shared/home/botond/utils/freesurfer/license.txt:/opt/freesurfer/license.txt:ro \
-v /shared/datasets/private/keck_bolus:/data:ro \
-v /shared/datasets/private/keck_bolus/derivatives:/out \
-v /shared/projects/scratch/botond:/scratch \
nipreps/fmriprep:20.2.3 /data /out \
participant \
--use-syn-sdc \
--ignore fieldmaps \
-w /scratch \
--participant-label sub-${sub}
