pip install -r requirements.txt

# install packages for chamferdist and pointnet2_ops_lib
git clone https://github.com/krrish94/chamferdist.git
cd chamferdist
git checkout 97051583f6fe72d5d4a855696dbfda0ea9b73a6a 
python setup.py install
cd ..

cd pointnet2_ops_lib
python setup.py install
cd ..