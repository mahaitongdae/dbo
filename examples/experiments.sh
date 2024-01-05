#for OBJ in 'bird' 'ackley' 'rosenbrock'
#do
#  python main_2d_json.py --acquisition_function 'es' --n_workers 5 --objective $OBJ
#  python main_2d_json.py --acquisition_function 'ucbpe' --n_workers 5 --objective $OBJ
#  python main_2d_json.py --acquisition_function 'bucb' --n_workers 5 --objective $OBJ
#  python main_2d_json.py --acquisition_function 'ei' --n_workers 5 --fantasies 5 --objective $OBJ
#  python main_2d_json.py --acquisition_function 'ts' --n_workers 5 --decision_type 'distributed' --objective $OBJ
#  python main_2d_json.py --acquisition_function 'es' --n_workers 5 --diversity_penalty True --objective $OBJ --div_radius 0.05
#  python main_2d_json.py --acquisition_function 'ei' --n_workers 10 --policy 'boltzmann' --decision_type 'distributed' --objective 'bird'
#  for RAD in 0.05 0.1 0.2
#  do
#    python main_2d_json.py --acquisition_function 'es' --n_workers 5 --diversity_penalty True --objective $OBJ --div_radius $RAD
#  done
#done
#python main_2d_json.py --acquisition_function 'ei' --n_workers 10 --policy 'boltzmann' --decision_type 'distributed' --objective 'bird'
#python main_2d_json.py --acquisition_function 'ei' --n_workers 10 --policy 'boltzmann' --decision_type 'distributed' --objective 'ackley'
#python main_2d_json.py --acquisition_function 'ei' --n_workers 10 --policy 'boltzmann' --decision_type 'distributed' --objective 'rosenbrock'
##python main_2d_json.py --acquisition_function 'es' --n_workers 5 --diversity_penalty True --objective 'bird' --div_radius 0.05
#python main_2d_json.py --acquisition_function 'es' --n_workers 5 --diversity_penalty True --objective 'ackley' --div_radius 0.05
#python main_2d_json.py --acquisition_function 'es' --n_workers 5 --diversity_penalty True --objective 'rosenbrock' --div_radius 0.05
#python main_2d_json.py --acquisition_function 'es' --n_workers 5 --diversity_penalty True --objective 'bird' --div_radius 0.1
#python main_2d_json.py --acquisition_function 'es' --n_workers 5 --diversity_penalty True --objective 'ackley' --div_radius 0.1
#python main_2d_json.py --acquisition_function 'es' --n_workers 5 --diversity_penalty True --objective 'rosenbrock' --div_radius 0.1
#python main_2d_json.py --acquisition_function 'es' --n_workers 5 --diversity_penalty True --objective 'bird' --div_radius 0.02
#python main_2d_json.py --acquisition_function 'es' --n_workers 5 --diversity_penalty True --objective 'ackley' --div_radius 0.02
#python main_2d_json.py --acquisition_function 'es' --n_workers 5 --diversity_penalty True --objective 'rosenbrock' --div_radius 0.02
#for OBJ in 'ackley' 'bird' 'rosenbrock'
#do
#  python main_2d_json.py --acquisition_function 'es' --n_workers 50 --objective $OBJ
#  python main_2d_json.py --acquisition_function 'ucbpe' --n_workers 50 --objective $OBJ
#  python main_2d_json.py --acquisition_function 'bucb' --n_workers 50 --objective $OBJ
#  python main_2d_json.py --acquisition_function 'ei' --n_workers 50 --fantasies 50 --objective $OBJ
#  python main_2d_json.py --acquisition_function 'ts' --n_workers 50 --objective $OBJ --random_search 100
#  python main_2d_json.py --acquisition_function 'sp' --n_workers 50 --objective $OBJ
#done

for OBJ in 'ackley' 'rosenbrock'
do
#  python main_2d_json.py --acquisition_function 'es' --n_workers 10 --objective $OBJ
#  python main_2d_json.py --acquisition_function 'ucbpe' --n_workers 10 --objective $OBJ
  python main_2d_json.py --acquisition_function 'bucb' --n_workers 10 --objective $OBJ
#  python main_2d_json.py --acquisition_function 'ei' --n_workers 50 --fantasies 50 --objective $OBJ
#  python main_2d_json.py --acquisition_function 'ts' --n_workers 50 --objective $OBJ --random_search 100
#  python main_2d_json.py --acquisition_function 'sp' --n_workers 50 --objective $OBJ
done

#for OBJ in 'ackley' 'bird' 'rosenbrock'
#do
#  python main_2d_json.py --acquisition_function 'ts' --n_workers 10 --objective $OBJ
#done
