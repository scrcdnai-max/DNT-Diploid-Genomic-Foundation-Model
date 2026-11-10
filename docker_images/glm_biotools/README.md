# glm_biotools Docker Image

Bioinformatics toolbox for the diploid_glm pipeline.

## Installed tools

| Tool | Version |
|------|---------|
| htslib (bgzip, tabix, htsfile) | 1.24 |
| bcftools | 1.24 |
| samtools | 1.24 |
| bedtools | 2.31.1 |
| UCSC utilities (liftOver, bigWig*/bigBed*, twoBit*, faSize) | latest prebuilt (v503 as of 2026-10) |

The UCSC utilities are a curated subset (`UCSC_TOOLS` build-arg) of UCSC's prebuilt static
binaries, installed in `/usr/local/ucsc/bin` (on `PATH`, after `/usr/local/bin`). UCSC does not
version these downloads, so a rebuild picks up whatever release is current. Add a tool with
`--build-arg UCSC_TOOLS="... newTool"` — any name from
<https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/>.

## Pinned tag

The tag every pipeline script uses is pinned **once**, in [`image.env`](image.env)
(`GLM_BIOTOOLS_IMAGE=glm_biotools:1.2.0`). The runners in `preprocess/huggingface_dataset/` `source` it and expose the value as
`DOCKER_IMAGE` (still overridable per run); the v10 dataset tests read it for their `bgzip`/`tabix`
shims. After rebuilding the image, bump `image.env` — nothing else hardcodes the tag.

```bash
docker save glm_biotools:1.2.0 -o glm_biotools_1.2.0.tar   # copy the image to a host that lacks it
docker load -i glm_biotools_1.2.0.tar
```

## Build

```bash
cd docker_images/glm_biotools
docker build -t glm_biotools:1.2.0 .
```

To override a tool version at build time:

```bash
docker build \
  --build-arg HTSLIB_VERSION=1.24 \
  --build-arg BCFTOOLS_VERSION=1.24 \
  --build-arg SAMTOOLS_VERSION=1.24 \
  --build-arg BEDTOOLS_VERSION=2.31.1 \
  -t glm_biotools:1.2.0 .
```

## Verify

```bash
docker run --rm glm_biotools:1.2.0 bash -c "
  bgzip --version | head -1
  tabix --version | head -1
  bcftools --version | head -1
  samtools --version | head -1
  bedtools --version | head -1
  liftOver 2>&1 | head -1
"
```

## Usage

Run any installed tool directly:

```bash
docker run --rm -v /path/to/data:/data glm_biotools:1.2.0 bgzip -@4 /data/file.vcf
docker run --rm -v /path/to/data:/data glm_biotools:1.2.0 tabix -p vcf /data/file.vcf.gz
docker run --rm -v /path/to/data:/data glm_biotools:1.2.0 bcftools view /data/file.bcf
docker run --rm -v /path/to/data:/data glm_biotools:1.2.0 samtools view /data/file.bam
docker run --rm -v /path/to/data:/data glm_biotools:1.2.0 bedtools intersect -a /data/a.bed -b /data/b.bed
docker run --rm -v /path/to/data:/data glm_biotools:1.2.0 bigWigAverageOverBed /data/phyloP.bw /data/regions.bed /data/out.tab
docker run --rm -v /path/to/data:/data glm_biotools:1.2.0 liftOver /data/in.bed /data/hg19ToHg38.over.chain.gz /data/out.bed /data/unmapped.bed
```

Interactive shell:

```bash
docker run --rm -it -v /path/to/data:/data glm_biotools:1.2.0
```
