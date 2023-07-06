num=350
model_list='unet2plus'
encoder_list='r152' #r101'
opt_list='Adam' # AdamW'
loss_list='dice' #dicefocal'
aug_list='base2' 
sch_list='CosineAnnealingLR'

num_workers=2
batch_size=4
learning_rate='1e-2'
max_epoch='25'


for model in $model_list
do
    for encoder in $encoder_list
    do
        for opt in $opt_list
        do
            for loss in $loss_list
            do
                for aug in $aug_list
                do
                    for sch in $sch_list
                    do
                        for epoch in $max_epoch
                        do
                            echo "exp_num:$num , model:$model, encoder:$encoder, opt:$opt, loss:$loss, aug:$aug, lr_scheduler: $sch" #, mixed:True" #
                            exp_name="${num}_${model}_${encoder}_${opt}_${loss}_${aug}_${learning_rate}_resized1024" #_${sch}_copypaste(k=9)" #_${sch}" #_mixed" #_${lr}" # 

                            python custom_train.py\
                            --exp_name $exp_name\
                            --k 9\
                            --model $model\
                            --encoder $encoder\
                            --optimizer $opt\
                            --loss $loss\
                            --aug $aug\
                            --num_workers $num_workers\
                            --batch_size $batch_size\
                            --learning_rate $learning_rate\
                            --max_epoch $epoch
                        
                            num=`expr $num + 1`
                        done
                    done
                done
            done 
        done    
    done
done