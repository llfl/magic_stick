FROM imux/openpose
MAINTAINER i@imux.top

RUN git clone https://github.com/llfl/magic_stick && cd magic_stick && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make 

WORKDIR /openpose/magic_stick

RUN cd models && /openpose/magic_stick/models/getModels.sh

CMD [ "/bin/bash" ]