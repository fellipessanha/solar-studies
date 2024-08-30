#! /usr/bin/fish

cd inmet-data/

for file in (ls)
    unzip $file
end

for year in (seq 2000 2024)
    mkdir -p $year
    mv *$year.CSV $year 
end