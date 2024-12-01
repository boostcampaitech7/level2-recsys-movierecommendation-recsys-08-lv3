import src.data as data_module
import src.models as model_module
import src.train as train_module
from omegaconf import OmegaConf
import argparse
from src.utils import *
import ast

def main(args):

    # seed
    seed_everything(args)
    
    datatype = args.model_args[args.model].datatype

    if datatype == 'basic':
        args.predict=True
    # data load
    else:
        data_split_fn = getattr(data_module, f'{datatype}_data_split')  # e.g. basic_data_split()
    data_load_fn = getattr(data_module, f'{datatype}_data_load')  # e.g. basic_data_load()
    data = data_load_fn(args)

    if datatype == 'context':
        data_sideinfo = getattr(data_module,f'{datatype}_data_sideinfo')
        data_sidemerge = getattr(data_module, f'{datatype}_data_side_merge')
        sample_negative_items = getattr(data_module,'sample_negative_items')
        data = data_sideinfo(args,data)

    model_name = args.model
    print(model_name)
    print(args.predict)
    trainer_assigned = getattr(train_module, f'{model_name}Trainer')
    
    # data_ side
    if not args.predict:
        # train
        print('---split---')
        data= data_split_fn(args,data)

        if datatype=='context':
            print('---sideinf merging---')
            data['train'] = sample_negative_items(data['train'], args.seed, args.negative)
            data=data_sidemerge(args,data)

        trainer =trainer_assigned(args,data)
        trained_model = trainer.train_model()
        print('---end---')
    else:
        # context data
        if datatype=='context':
            data['total'] = sample_negative_items(data['total'], args.seed, args.negative_samples)
            data=data_sidemerge(args,data)
            trainer= trainer_assigned(args,data)
            trained_model = trainer.train_model()
            predict_base = trainer.generate_prediction_base()
            predictions = trainer.evaluate(trained_model, predict_base, top_k=10)
        else:
            trainer= trainer_assigned(args,data)
            trained_model= trainer.train()
            scores = trainer.predict()
            predictions = trainer.evaluate(scores.cpu().numpy())
        save_to_csv(args, data, predictions)
        
    
        

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='parser')

    arg = parser.add_argument

    arg('--config', '-c', '--c', type=str, 
        help='Configuration 파일을 설정합니다.', required=True)
    arg('--datapath',type=str,
        help='datapath를 지정하시오')
    arg('--model', '-m', '--m', type=str, 
        choices=['DeepFM'],
        help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--seed','-s','--s',type=int,
        help= '시드설정')
    arg('--device', '-d', '--d', type=str, 
         help='사용할 디바이스를 선택할 수 있습니다.')
    arg('--negative',type=int,help='negative sampling갯수를 지정합니다')
    arg('--dataloader', '--dl', '-dl', type=ast.literal_eval)
    arg('--model_args', '--ma', '-ma', type=ast.literal_eval)
    arg('--predict', '-p', '--p', '--pred', type=ast.literal_eval, 
        help='학습을 생략할지 여부를 설정할 수 있습니다.')
   
    
    args = parser.parse_args()
    
    config_args = OmegaConf.create(vars(args))
    config_yaml = OmegaConf.load(args.config) if args.config else OmegaConf.create()

    for key in config_args.keys():
        if config_args[key] is not None:
            config_yaml[key] = config_args[key]
    # 사용되지 않는 정보 삭제 (학습 시에만)
    
    # Configuration 콘솔에 출력
    print(OmegaConf.to_yaml(config_yaml))

    main(config_yaml)