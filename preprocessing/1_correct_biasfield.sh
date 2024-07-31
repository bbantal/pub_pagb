# Get inputs
read -p "Enter subject ID [XXX format]: " SUB
read -p "Enter sessions [eg. bhb glc]: " SESSIONS

# Load matlab module
module load matlab

# Run python script
python preprocessing/correct_biasfield.py "${SUB}" "${SESSIONS}"

# Clean up
rm pyscript_segment.m
