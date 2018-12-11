# !/bin/bash
nepoch=300
for datapath in wiki_data_timestep_100; do
	for k in 100;do
		for margin in 1;do
			for rate in 0.01; do
				for coreNum in 1; do
					resultpath="./"$datapath"/HyTE_"$coreNum"thread_k"$k"_margin"$margin"_epoch"$nepoch"_rate"$rate
					
					if [ -d $resultpath ]; then
						rm -r $resultpath
					fi
					if [ ! -d $resultpath ]; then
						echo "mkdir "$resultpath
						mkdir $resultpath
						./ParHyTE unif $coreNum $datapath $k $margin $nepoch $resultpath $rate HyTE_split > $resultpath/train.txt
						
						if [ $? -eq 0 ];then
						    ./ParTest_HyTE unif $coreNum $datapath $k $margin $nepoch $resultpath $rate HyTE_split > $resultpath/test.txt
						else
						    exit
						fi

						if [ $? -eq 0 ];then
							continue
						else 
						    exit
						fi
					fi
				done
			done
		done		
	done
done


