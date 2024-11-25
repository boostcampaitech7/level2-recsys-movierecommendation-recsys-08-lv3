import argparse
from sklearn import train_valid_split

def main():
    data= train_valid_split()
    train(data)
    args.model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='parser')

    arg = parser.add_argument
   # arg('--config', '-c', '--c', type=str, 
    #    help='Configuration 파일을 설정합니다.', required=True)
    arg('--datapath',type=str,
        help='datapath를 지정하시오')
    arg('--model', '-m', '--m', type=str, 
        choices=['catboost', 'DeepFM','MF','NeuralMF'],
        help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--seed','-s','--s',type=int,
        help= '시드설정')
    arg('--device', '-d', '--d', type=str, 
        help='사용할 디바이스를 선택할 수 있습니다.')
    arg('--negative',type=int,help='negative sampling갯수를 지정합니다')
    #arg('--dataloader', '--dl', '-dl', type=ast.literal_eval)
    #arg('--dataset', '--dset', '-dset', type=ast.literal_eval)
    #arg('--optimizer', '-opt', '--opt', type=ast.literal_eval)
    #arg('--loss', '-l', '--l', type=str)
    #arg('--metrics', '-met', '--met', type=ast.literal_eval)
    #arg('--train', '-t', '--t', type=ast.literal_eval)
    
    args = parser.parse_args()
    args.datapath=''
    args.seed=42
    args.device='cuda'
    