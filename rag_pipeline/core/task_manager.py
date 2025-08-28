import asyncio
import uuid
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..api.models import JobStatus, DocumentIngestionResult, BulkJobStatus, DocumentIngestionItem

logger = logging.getLogger(__name__)


@dataclass
class BulkIngestionJob:
    job_id: str
    documents: List[DocumentIngestionItem]
    batch_name: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    results: List[DocumentIngestionResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    processed_count: int = 0
    successful_count: int = 0
    failed_count: int = 0


class BulkIngestionTaskManager:
    def __init__(self, max_concurrent_jobs: int = 3, max_concurrent_docs_per_job: int = 5):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_concurrent_docs_per_job = max_concurrent_docs_per_job
        self.jobs: Dict[str, BulkIngestionJob] = {}
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self.job_semaphore = asyncio.Semaphore(max_concurrent_jobs)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs * 2)
    
    def create_job(
        self, 
        documents: List[DocumentIngestionItem], 
        batch_name: Optional[str] = None
    ) -> str:
        job_id = f"bulk_{uuid.uuid4()}"
        
        job = BulkIngestionJob(
            job_id=job_id,
            documents=documents,
            batch_name=batch_name,
            results=[
                DocumentIngestionResult(
                    url=str(doc.url),
                    filename=doc.filename,
                    status=JobStatus.PENDING
                )
                for doc in documents
            ]
        )
        
        self.jobs[job_id] = job
        logger.info(f"Created bulk ingestion job {job_id} with {len(documents)} documents")
        return job_id
    
    async def start_job(
        self, 
        job_id: str, 
        ingestion_callback: Callable[[str, str, Optional[Dict[str, Any]]], Any]
    ):
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        
        if job_id in self.active_jobs:
            raise ValueError(f"Job {job_id} is already running")
        
        job = self.jobs[job_id]
        task = asyncio.create_task(self._process_job(job, ingestion_callback))
        self.active_jobs[job_id] = task
        
        # Don't await here - we want to return immediately
        logger.info(f"Started background processing for job {job_id}")
    
    async def _process_job(
        self, 
        job: BulkIngestionJob, 
        ingestion_callback: Callable[[str, str, Optional[Dict[str, Any]]], Any]
    ):
        async with self.job_semaphore:
            try:
                job.status = JobStatus.PROCESSING
                job.started_at = datetime.now(timezone.utc)
                logger.info(f"Processing job {job.job_id} with {len(job.documents)} documents")
                
                # Process documents in batches with concurrency control
                semaphore = asyncio.Semaphore(self.max_concurrent_docs_per_job)
                tasks = []
                
                for i, doc in enumerate(job.documents):
                    task = asyncio.create_task(
                        self._process_single_document(
                            job, i, doc, ingestion_callback, semaphore
                        )
                    )
                    tasks.append(task)
                
                # Wait for all documents to be processed
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Update job completion status
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                
                logger.info(
                    f"Completed job {job.job_id}: "
                    f"{job.successful_count} successful, {job.failed_count} failed"
                )
                
            except Exception as e:
                logger.error(f"Job {job.job_id} failed: {e}")
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                job.completed_at = datetime.now(timezone.utc)
            
            finally:
                # Clean up the active job reference
                if job.job_id in self.active_jobs:
                    del self.active_jobs[job.job_id]
    
    async def _process_single_document(
        self,
        job: BulkIngestionJob,
        doc_index: int,
        doc: DocumentIngestionItem,
        ingestion_callback: Callable[[str, str, Optional[Dict[str, Any]]], Any],
        semaphore: asyncio.Semaphore
    ):
        async with semaphore:
            start_time = time.time()
            result = job.results[doc_index]
            
            try:
                result.status = JobStatus.PROCESSING
                logger.debug(f"Processing document {doc.filename} in job {job.job_id}")
                
                # Call the ingestion function
                document_id = await ingestion_callback(
                    str(doc.url), 
                    doc.filename, 
                    doc.metadata
                )
                
                # Update result
                result.status = JobStatus.COMPLETED
                result.document_id = document_id
                result.processing_time_seconds = time.time() - start_time
                
                job.successful_count += 1
                logger.debug(f"Successfully processed {doc.filename} -> {document_id}")
                
            except Exception as e:
                logger.error(f"Failed to process {doc.filename} in job {job.job_id}: {e}")
                result.status = JobStatus.FAILED
                result.error_message = str(e)
                result.processing_time_seconds = time.time() - start_time
                
                job.failed_count += 1
            
            finally:
                job.processed_count += 1
    
    def get_job_status(self, job_id: str) -> Optional[BulkJobStatus]:
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        
        return BulkJobStatus(
            job_id=job.job_id,
            status=job.status,
            batch_name=job.batch_name,
            total_documents=len(job.documents),
            processed_documents=job.processed_count,
            successful_documents=job.successful_count,
            failed_documents=job.failed_count,
            results=job.results,
            created_at=job.created_at.isoformat(),
            started_at=job.started_at.isoformat() if job.started_at else None,
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            error_message=job.error_message
        )
    
    def list_jobs(self, limit: int = 50) -> List[BulkJobStatus]:
        jobs = sorted(
            self.jobs.values(), 
            key=lambda j: j.created_at, 
            reverse=True
        )[:limit]
        
        return [
            BulkJobStatus(
                job_id=job.job_id,
                status=job.status,
                batch_name=job.batch_name,
                total_documents=len(job.documents),
                processed_documents=job.processed_count,
                successful_documents=job.successful_count,
                failed_documents=job.failed_count,
                results=job.results,
                created_at=job.created_at.isoformat(),
                started_at=job.started_at.isoformat() if job.started_at else None,
                completed_at=job.completed_at.isoformat() if job.completed_at else None,
                error_message=job.error_message
            )
            for job in jobs
        ]
    
    def cancel_job(self, job_id: str) -> bool:
        if job_id in self.active_jobs:
            task = self.active_jobs[job_id]
            task.cancel()
            del self.active_jobs[job_id]
            
            if job_id in self.jobs:
                self.jobs[job_id].status = JobStatus.FAILED
                self.jobs[job_id].error_message = "Job cancelled by user"
                self.jobs[job_id].completed_at = datetime.now(timezone.utc)
            
            logger.info(f"Cancelled job {job_id}")
            return True
        
        return False
    
    def cleanup_completed_jobs(self, max_age_hours: int = 24):
        cutoff_time = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        
        jobs_to_remove = []
        for job_id, job in self.jobs.items():
            if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED] and 
                job.created_at.timestamp() < cutoff_time):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.jobs[job_id]
            logger.info(f"Cleaned up old job {job_id}")
        
        return len(jobs_to_remove)
    
    async def shutdown(self):
        logger.info("Shutting down bulk ingestion task manager")
        
        # Cancel all active jobs
        for job_id, task in self.active_jobs.items():
            task.cancel()
            logger.info(f"Cancelled job {job_id} during shutdown")
        
        # Wait for all tasks to complete or be cancelled
        if self.active_jobs:
            await asyncio.gather(*self.active_jobs.values(), return_exceptions=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Bulk ingestion task manager shutdown complete")