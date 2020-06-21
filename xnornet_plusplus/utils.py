import time, logging, os
from tensorboardX import SummaryWriter
def create_logger(cfg, phase='train'):
    if not os.path.exists(cfg.model_dir):
        os.makedirs(cfg.model_dir)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    log_file = os.path.join(cfg.model_dir,log_file)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    #formatter = logging.Formatter('%(asctime)s %(message)s')
    handler = logging.StreamHandler() #for normal printing in the terminal
    
    handler.setLevel(logging.DEBUG)
    #handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    handler = logging.FileHandler(log_file) #for normal printing in the terminal
    handler.setLevel(logging.DEBUG)
    #handler.setFormatter(formatter)
    logger.addHandler(handler)

    tb_log_dir = os.path.join(cfg.model_dir,"tensorboard")
    writer = SummaryWriter(tb_log_dir)

    return writer
