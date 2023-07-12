FROM fairembodied/habitat-challenge:homerobot-ovmm-challenge-2023

# install baseline agent requirements
RUN /bin/bash -c "\
    . activate home-robot \
    && cd home-robot \
    && git submodule update --init --recursive src/third_party/detectron2 \
        src/home_robot/home_robot/perception/detection/detic/Detic \
        src/third_party/contact_graspnet \
    && pip install -e src/third_party/detectron2 \
    && pip install -r src/home_robot/home_robot/perception/detection/detic/Detic/requirements.txt \
    "

# download pretrained checkpoint
RUN mkdir -p home-robot/src/home_robot/home_robot/perception/detection/detic/Detic/models && \
    wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        -O home-robot/src/home_robot/home_robot/perception/detection/detic/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        --no-check-certificate

# add baseline agent code
ADD eval_baselines_agent.py /home-robot/projects/habitat_ovmm/agent.py

# add submission script
ADD scripts/submission.sh /home-robot/submission.sh

# set evaluation type to remote
ENV AGENT_EVALUATION_TYPE remote

# run submission script
CMD /bin/bash -c "\
    . activate home-robot \
    && cd /home-robot \
    && export PYTHONPATH=/evalai_remote_evaluation:$PYTHONPATH \
    && bash submission.sh \
    "
