FROM fairembodied/habitat-challenge:homerobot-ovmm-challenge-2023-dev2

RUN /bin/bash -c "\
    . activate home-robot && \
    cd home-robot && \
    git submodule update --init --recursive src/third_party/detectron2 \
        src/home_robot/home_robot/perception/detection/detic/Detic \
        src/third_party/contact_graspnet && \
    pip install -e src/third_party/detectron2 && \
    pip install -r src/home_robot/home_robot/perception/detection/detic/Detic/requirements.txt \
    "

RUN mkdir -p home-robot/src/home_robot/home_robot/perception/detection/detic/Detic/models && \
    wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        -O home-robot/src/home_robot/home_robot/perception/detection/detic/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        --no-check-certificate

ADD eval_baselines_agent.py /home-robot/projects/habitat_ovmm/agent.py
ADD scripts/submission.sh /home-robot/submission.sh

CMD ["/bin/bash", "-c", ". activate home-robot; cd /home-robot; bash submission.sh"]
