

addimport () {
	if ! pcregrep -qM "from . import $1" ./__init__.py; then
       		printf  "\nfrom . import $1" >> ./__init__.py
	fi
}
export -f addimport

#ls | grep -x "[^_][^ ]*.py" | sed "s/.py//" | xargs bash -t -c 'addimport "$@"' _


ls | grep -x "[^_][^ ]*.py" | sed "s/.py//" | parallel -j1 addimport {} 

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
pip install gym -y
pip install pyhash -y 


