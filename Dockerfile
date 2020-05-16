FROM imux/openpose
MAINTAINER i@imux.top

RUN git clone https://github.com/llfl/magic_stick && cd magic_stick && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make && cd ..

RUN /openpose/magic_stick/models/getModels.sh

WORKDIR /openpose/magic_stick

CMD [ "/bin/bash" ]