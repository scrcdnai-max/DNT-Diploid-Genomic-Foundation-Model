FROM ubuntu:22.04

LABEL maintainer="scrc.dnai@gmail.com"
LABEL description="GLM bioinformatics toolbox"
LABEL version="1.2.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/bin:/usr/local/ucsc/bin:${PATH}"

# Build deps shared by htslib / bcftools / samtools / bedtools
RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf automake bzip2 ca-certificates curl g++ gcc \
    libbz2-dev libcurl4-openssl-dev liblzma-dev \
    libncurses5-dev libssl-dev make perl python3 wget zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# htslib 1.24 — provides bgzip, tabix, htsfile (bcftools/samtools bundle their
# own htslib copy but don't install these CLIs). Keep in lockstep with bcftools/samtools.
ARG HTSLIB_VERSION=1.24
RUN wget -q "https://github.com/samtools/htslib/releases/download/${HTSLIB_VERSION}/htslib-${HTSLIB_VERSION}.tar.bz2" \
    && tar -xjf htslib-${HTSLIB_VERSION}.tar.bz2 \
    && cd htslib-${HTSLIB_VERSION} \
    && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig \
    && cd .. \
    && rm -rf htslib-${HTSLIB_VERSION} htslib-${HTSLIB_VERSION}.tar.bz2

# bcftools 1.24 (bundled htslib)
ARG BCFTOOLS_VERSION=1.24
RUN wget -q "https://github.com/samtools/bcftools/releases/download/${BCFTOOLS_VERSION}/bcftools-${BCFTOOLS_VERSION}.tar.bz2" \
    && tar -xjf bcftools-${BCFTOOLS_VERSION}.tar.bz2 \
    && cd bcftools-${BCFTOOLS_VERSION} \
    && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install \
    && cd .. \
    && rm -rf bcftools-${BCFTOOLS_VERSION} bcftools-${BCFTOOLS_VERSION}.tar.bz2

# samtools 1.24 (bundled htslib)
ARG SAMTOOLS_VERSION=1.24
RUN wget -q "https://github.com/samtools/samtools/releases/download/${SAMTOOLS_VERSION}/samtools-${SAMTOOLS_VERSION}.tar.bz2" \
    && tar -xjf samtools-${SAMTOOLS_VERSION}.tar.bz2 \
    && cd samtools-${SAMTOOLS_VERSION} \
    && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install \
    && cd .. \
    && rm -rf samtools-${SAMTOOLS_VERSION} samtools-${SAMTOOLS_VERSION}.tar.bz2

# bedtools 2.31.1
ARG BEDTOOLS_VERSION=2.31.1
RUN wget -q "https://github.com/arq5x/bedtools2/releases/download/v${BEDTOOLS_VERSION}/bedtools-${BEDTOOLS_VERSION}.tar.gz" \
    && tar -xzf bedtools-${BEDTOOLS_VERSION}.tar.gz \
    && cd bedtools2 \
    && make -j"$(nproc)" \
    && cp bin/* /usr/local/bin/ \
    && cd .. \
    && rm -rf bedtools2 bedtools-${BEDTOOLS_VERSION}.tar.gz

# UCSC Genome Browser utilities ("kent"): a curated subset of the prebuilt static
# binaries. UCSC only publishes the latest build here (unversioned, overwritten each
# release — v503 as of 2026-10); the pinned userApps.v503 source tarball is broken
# (missing checkCompileFlags.sh), so we don't build from source.
ARG UCSC_TOOLS="liftOver bigWigToBedGraph bedGraphToBigWig bigWigAverageOverBed \
    bigWigInfo bigWigSummary bigWigToWig wigToBigWig bedToBigBed bigBedToBed \
    bigBedInfo faToTwoBit twoBitToFa twoBitInfo faSize"
RUN mkdir -p /usr/local/ucsc/bin \
    && for t in ${UCSC_TOOLS}; do \
         wget -q -O /usr/local/ucsc/bin/$t "https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/$t" || exit 1; \
       done \
    && chmod +x /usr/local/ucsc/bin/*

CMD ["/bin/bash"]
