# IBM AI Workflow Capstone Project

https://github.com/aavail/ai-workflow-capstone

## Tests
To run all tests at once:
```python 
python -m pytest 
```
To run specific tests:
```python 
python -m pytest tests/{filename}
```
where `{filename}` can be `test_api.py`, `test_logger.py` or `test_model.py`.

## Scripts

The `scripts` package contains all the useful modules to build datasets and models.

## Docker
To build a Docker image:
```bash
docker build -t image_name .
```
To see Docker images:
```bash
docker images
```
To run it:
```bash
docker run -p 4000:8080 image_name
```
then go to http://0.0.0.0:4000/.

## Monitor

To obtain a monitore of the performance on production data, run the python script monitoring 

```bash
python scripts.monitoring.py -c None
```