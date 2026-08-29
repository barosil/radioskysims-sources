# interfaces/api/satellite_controller.py
# Acho que não precisa de FASTAPI, SQLModel seria o suficiente.
from __future__ import annotations

from fastapi import APIRouter, HTTPException


def router(service):
    router = APIRouter()

    @router.get("/")
    def list_satellites():
        return service.get_all_satellites()

    @router.get("/{satellite_id}")
    def get_satellite(satellite_id: int):
        sat = service.get_satellite_by_id(satellite_id)
        if sat is None:
            raise HTTPException(status_code=404, detail="Satellite not found")
        return sat

    @router.post("/")
    def add_satellite(satellite_data: dict):
        return service.add_satellite(satellite_data)

    @router.put("/{satellite_id}")
    def update_satellite(satellite_id: int, update_data: dict):
        return service.update_satellite(satellite_id, update_data)

    @router.delete("/{satellite_id}")
    def delete_satellite(satellite_id: int):
        return service.delete_satellite(satellite_id)

    return router
