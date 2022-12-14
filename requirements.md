# libraries
```
sudo apt install librange-v3-dev
```

# datasets

## N-MNist

```
mkdir datasets
cd datasets
  wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/468j46mzdv-1.zip
  unzip 468j46mzdv-1.zip
  rm 468j46mzdv-1.zip
  mv 468j46mzdv-1 n-mnist
  cd n-mnist
    unzip Test.zip
    unzip Train.zip
    rm Test.zip
    rm Train.zip
  cd ..
```

# c++
Tested on Debian/Linux 11 using g++ version 10.2.1 with `-std=c++2a -fconcepts` options.

