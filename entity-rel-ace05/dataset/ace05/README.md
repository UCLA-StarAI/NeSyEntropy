# ACE2005 preprocessing

This is a simple code for preprocessing ACE 2005 corpus for Event Extraction task. 

Using the existing methods were complicated for me, so I made this project.

## Prerequisites

1. Prepare **ACE 2005 dataset**. 

   (Download: https://catalog.ldc.upenn.edu/LDC2006T06. Note that ACE 2005 dataset is not free.)

2. Install the packages.
   ```
   pip install stanfordcorenlp beautifulsoup4 nltk tqdm
   ```
    
3. Download stanford-corenlp model.
    ```bash
    wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
    unzip stanford-corenlp-full-2018-10-05.zip
    ```

## Usage

Run:

```bash
sudo python main.py --data=./data/ace_2005_td_v7/data/English
``` 

- Then you can get the parsed data in `output directory`. 

- If it is not executed with the `sudo`, an error can occur when using `stanford-corenlp`.

- It takes about 30 minutes to complete the pre-processing.

```bash
cd output
python ace_to_tacred.py
python fix_ner.py
cp *.json ..
``` 

### Data Split

The result of data is divided into test/dev/train as follows.
```
├── output
│     └── test.json
│     └── dev.json
│     └── train.json
│...
```

This project use the same data partitioning as the previous work ([Yang and Mitchell, 2016](https://www.cs.cmu.edu/~bishan/papers/joint_event_naacl16.pdf);  [Nguyen et al., 2016](https://www.aclweb.org/anthology/N16-1034)). The data segmentation is specified in `data_list.csv`.

Below is information about the amount of parsed data when using this project. It is slightly different from the parsing results of the two papers above. The difference seems to have occurred because there are no promised rules for splitting sentences within the sgm format files.

|          | Documents    |  Sentences   |Triggers    | Arguments | Entity Mentions  |
|-------   |--------------|--------------|------------|-----------|----------------- |
| Test     | 40        | 713           | 422           | 892             |  4226             |
| Dev      | 30        | 875           | 492           | 933             |  4050             |
| Train    | 529       | 14724         | 4312          | 7811             |   53045            |
