import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from utils.visualize import vis_utils
from glob import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, )
    # parser.add_argument("--out",type=str,required=True,help="results save folder")
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    params = parser.parse_args()


    
    # params.file = '/home/deli/project/momask-codes/visualize/013.npy'
    if params.folder is not None:
        files = sorted(glob(os.path.join(params.folder, '*.npy')))
    elif params.file is not None:
        files = [params.file]

    for file in files:
        npy_path = file
        out_npz_path = file.replace('.npy', '.npz')
        if os.path.exists(out_npz_path):
            continue

        print(f'Processing {out_npz_path}')

        assert os.path.exists(npy_path)
        params.out = os.path.join(os.path.dirname(npy_path),"export_objs")
        if not os.path.exists(params.out):
            os.mkdir(params.out)

        npy2obj = vis_utils.npy2obj(npy_path,device=params.device, cuda=params.cuda)

        # print(f'Saving obj files to {params.out}')
        
        #save obj
        # for frame_i in tqdm(range(npy2obj.real_num_frames)):
        #     npy2obj.save_obj(os.path.join(params.out, 'frame{:03d}.obj'.format(frame_i)), frame_i)

        print(f'Saving SMPL params npy file to {out_npz_path}')

        # npy2obj.save_npy(out_npy_path)
        npy2obj.save_npz(out_npz_path)