from dataclasses import dataclass

import ray

import torch

import pixeltable as pxt
from pixeltable.catalog import Table
from pixeltable.exprs import Expr

from goldener.pxt_utils import get_expr_from_column_name


def start_ray_cluster(
    address: None | str = None,
    num_cpus: int | None = None,
    num_gpus: int | None = None,
) -> None:
    """Start a Ray server.

    This is assuming that the user has already configured an existing Ray server with defined cluster.
    In that case the adress is required to be provided.

    Otherwise, this is assuming that this is a test setup ran locally, in which case the number of CPUs and GPUs can be specified.

    Args:
        address: The address of the Ray server to connect to.
        num_cpus: The number of CPUs to allocate for the local Ray instance.
        num_gpus: The number of GPUs to allocate for the local Ray instance.
    """
    if address:
        ray.init(address=address)
    else:
        ray.init(num_cpus=num_cpus, num_gpus=num_gpus)


@dataclass
class CudaDeviceInfo:
    """Information about a CUDA device.

    Attributes:
        id: The device ID.
        name: The device name.
        total_memory_gb: The total memory of the device in GB.
        free_memory_gb: The free memory of the device in GB.
    """

    id: int
    name: str
    total_memory_gb: float
    free_memory_gb: float


def get_cuda_device_info(cuda_id: int) -> CudaDeviceInfo:
    torch.cuda.set_device(cuda_id)
    props = torch.cuda.get_device_properties()

    total_memory_gb = round(props.total_memory / (1024**3), 2)
    used_memory_gb = round(torch.cuda.device_memory_used() / (1024**3), 2)

    free_memory_gb = total_memory_gb - used_memory_gb

    return CudaDeviceInfo(
        id=cuda_id,
        name=props.name,
        total_memory_gb=total_memory_gb,
        free_memory_gb=free_memory_gb,
    )


def get_cuda_device_infos() -> list[CudaDeviceInfo]:
    """get the CudaDeviceInformation for all cuda devices of the machine."""
    return [
        get_cuda_device_info(cuda_id) for cuda_id in range(torch.cuda.device_count())
    ]


@dataclass
class RayNodeInfo:
    """Information about a Ray node.

    Attributes:
        alive: Whether the node is alive.
        node_id: The ID of the node.
        node_name: The name of the node.
        node_manager_address: The address of the node manager.
        node_manager_hostname: The hostname of the node manager.
        num_cpus: The number of CPUs on the node.
        ram_gb: The amount of RAM on the node in GB.
        cuda_device_infos: A list of CudaDeviceInfo objects for the node, or None if the node has no GPUs or is not alive.
    """

    alive: bool
    node_id: str
    node_name: str
    node_manager_address: str
    node_manager_hostname: str
    num_cpus: int
    ram_gb: float
    cuda_device_infos: list[CudaDeviceInfo] | None


def get_ray_cluster_info() -> list[RayNodeInfo]:
    """Get information about all nodes in the Ray cluster."""
    remote_cuda_vram = ray.remote(get_cuda_device_infos)
    cuda_vram_futures = {}

    infos = []
    for node in ray.nodes():
        resources = node["Resources"]
        alive = node["Alive"]
        node_name = node["NodeName"]
        node_id = node["NodeID"]

        infos.append(
            RayNodeInfo(
                alive=alive,
                node_id=node_id,
                node_name=node_name,
                node_manager_address=node["NodeManagerAddress"],
                node_manager_hostname=node["NodeManagerHostname"],
                num_cpus=int(resources.get("CPU", 0)),
                ram_gb=round(resources.get("memory", 0) / (1024**3), 2),
                cuda_device_infos=None,  # to be filled later if GPU is present and node alive
            )
        )

        gpu_count = int(resources.get("GPU", 0))
        # If node has GPU, query cuda infos via remote tasks for each node
        if alive and gpu_count > 0:
            node_link = f"node:{node_name}"
            cuda_vram_futures[node_id] = remote_cuda_vram.options(
                num_gpus=gpu_count,
                resources={
                    node_link: 0.01,
                },
            ).remote()

    # Gather cuda device infos for all node with GPUs
    for node_info in infos:
        if node_info.node_id in cuda_vram_futures:
            node_info.cuda_device_infos = ray.get(cuda_vram_futures[node_info.node_id])

    return infos


@dataclass
class TorchModelCudaUsageInfo(CudaDeviceInfo):
    """Information about CUDA memory usage on a device.

    Attributes:
        id: The device ID.
        model_memory_gb: The amount of memory used by the model on the device in GB.
        call_memory_gb: The amount of memory used by the model call on the device in GB.

        See CudaDeviceInfo for other attributes.
    """

    model_memory_gb: float
    call_memory_gb: float


def get_cuda_device_usage_info(
    cuda_id: int,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> TorchModelCudaUsageInfo:
    cuda_device_info = get_cuda_device_info(cuda_id)

    device = torch.device(f"cuda:{cuda_id}")
    torch.cuda.reset_peak_memory_stats(None)
    model = model.to(device)
    model_memory_gb = round(torch.cuda.max_memory_allocated(device) / (1024**3), 6)

    torch.cuda.reset_peak_memory_stats(device)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        _ = model(input_tensor)
    call_memory_gb = round(torch.cuda.max_memory_allocated(device) / (1024**3), 6)

    return TorchModelCudaUsageInfo(
        id=cuda_id,
        name=cuda_device_info.name,
        total_memory_gb=cuda_device_info.total_memory_gb,
        free_memory_gb=cuda_device_info.free_memory_gb,
        model_memory_gb=model_memory_gb,
        call_memory_gb=call_memory_gb,
    )


def get_cuda_device_usage_infos(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> list[TorchModelCudaUsageInfo]:
    """Get information about CUDA memory usage for a model and an input tensor."""
    return [
        get_cuda_device_usage_info(cuda_id, model, input_tensor)
        for cuda_id in range(torch.cuda.device_count())
    ]


@dataclass
class RayNodeCudaUsageInfo:
    """Information about CUDA memory usage on a Ray node.

    Attributes:
        node_id: The ID of the node.
        node_name: The name of the node.
        cuda_usage_infos: A list of TorchModelCudaUsageInfo objects for the node.
    """

    node_id: str
    node_name: str
    cuda_usage_infos: list[TorchModelCudaUsageInfo] | None


def get_ray_cluster_cuda_usage_infos(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> list[RayNodeCudaUsageInfo]:
    """Get information about all nodes in the Ray cluster."""
    remote_cuda_usage = ray.remote(get_cuda_device_usage_infos)
    cuda_usage_futures = {}

    infos = []
    for node in ray.nodes():
        resources = node["Resources"]
        node_id = node["NodeID"]
        node_name = node["NodeName"]

        infos.append(
            RayNodeCudaUsageInfo(
                node_id=node_id,
                node_name=node_name,
                cuda_usage_infos=None,  # to be filled later if GPU is present and node alive
            )
        )

        gpu_count = int(resources.get("GPU", 0))
        # If node has GPU, query cuda infos via remote tasks for each node
        if node["Alive"] and gpu_count > 0:
            node_link = f"node:{node_name}"
            cuda_usage_futures[node_id] = remote_cuda_usage.options(
                num_gpus=gpu_count,
                resources={
                    node_link: 1,
                },
            ).remote(model, input_tensor)

    # Gather cuda device infos for all node with GPUs
    for node_info in infos:
        if node_info.node_id in cuda_usage_futures:
            node_info.cuda_usage_infos = ray.get(cuda_usage_futures[node_info.node_id])

    return infos


@dataclass
class CudaDeviceFreeMemoryInfo:
    """Assignments of rows to a CUDA device.

    Attributes:
        cuda_device_id: The CUDA device ID.
    """

    node_id: str
    node_name: str
    cuda_device_id: int
    call_memory_gb: float
    free_memory_call_gb: float
    ray_nodes_ratio: float = 0.0


def get_ray_cluster_cuda_free_memory_infos(
    node_cuda_usage_infos: list[RayNodeCudaUsageInfo],
    minimum_free_memory_gb: float | None = None,
) -> list[CudaDeviceFreeMemoryInfo]:
    cuda_device_free_memory_infos: list[CudaDeviceFreeMemoryInfo] = []

    for node_cuda_usage_info in node_cuda_usage_infos:
        if node_cuda_usage_info.cuda_usage_infos is None:
            continue

        for cuda_device_usage_info in node_cuda_usage_info.cuda_usage_infos:
            if (
                minimum_free_memory_gb is not None
                and cuda_device_usage_info.free_memory_gb < minimum_free_memory_gb
            ):
                continue

            cuda_device_free_memory_infos.append(
                CudaDeviceFreeMemoryInfo(
                    node_name=node_cuda_usage_info.node_name,
                    node_id=node_cuda_usage_info.node_id,
                    cuda_device_id=cuda_device_usage_info.id,
                    call_memory_gb=cuda_device_usage_info.call_memory_gb,
                    free_memory_call_gb=cuda_device_usage_info.free_memory_gb
                    - cuda_device_usage_info.model_memory_gb,
                )
            )

    total_call_free_memory_gb = sum(
        info.free_memory_call_gb for info in cuda_device_free_memory_infos
    )

    for info in cuda_device_free_memory_infos:
        info.ray_nodes_ratio = info.free_memory_call_gb / total_call_free_memory_gb

    return cuda_device_free_memory_infos


def assign_rows_to_ray_cuda_devices(
    pxt_table: Table,
    node_cuda_usage_infos: list[RayNodeCudaUsageInfo],
    col_name: str,
    minimum_free_memory_gb: float | None = None,
) -> None:
    """Assign rows to CUDA devices in order to distribute workload.

    It will add a new column to the table with the name `col_name` and assign each row a value corresponding to a value.

    Args:
        pxt_table: the table to distribute compute on.
        node_cuda_usage_infos: All Ray nodes with their CUDA usage information.
        col_name: The name of the column used to assign rows.
    """
    pxt_table.add_column(if_exists="error", **{col_name: pxt.Json})
    column_expr = get_expr_from_column_name(pxt_table, col_name)

    cuda_device_free_memory_infos = get_ray_cluster_cuda_free_memory_infos(
        node_cuda_usage_infos, minimum_free_memory_gb
    )

    total_rows = pxt_table.count()
    for info_idx, cuda_device_free_memory_info in enumerate(
        cuda_device_free_memory_infos, 1
    ):
        if info_idx == len(cuda_device_free_memory_infos):
            # assign all remaining rows to the last device to avoid rounding issues
            row_count = pxt_table.where(column_expr == None).count()  # noqa: E711
        else:
            row_count = int(total_rows * cuda_device_free_memory_info.ray_nodes_ratio)

        assigned_idx = [
            row["idx"]
            for row in pxt_table.where(column_expr == None)  # noqa: E711
            .select(pxt_table.idx)
            .sample(row_count)
            .collect()
        ]
        pxt_table.where(pxt_table.idx.isin(assigned_idx)).update(
            {
                col_name: {
                    "node_id": cuda_device_free_memory_info.node_id,
                    "node_name": cuda_device_free_memory_info.node_name,
                    "cuda_device_id": cuda_device_free_memory_info.cuda_device_id,
                }
            }
        )


def get_pxt_torch_dataset_for_ray_cuda_device(
    pxt_table: Table,
    col: Expr,
    node_id: str,
    cuda_device_id: int,
    select_exprs: list[Expr] | None = None,
) -> torch.utils.data.IterableDataset:
    """Get a PyTorch dataset for a specific Ray node and CUDA device.

    Args:
        pxt_table: The PixelTable table.
        col: The expression object for the column used to assign rows.
        node_id: The ID of the Ray node.
        cuda_device_id: The ID of the CUDA device.
        select_exprs: Optional list of expressions to select specific columns. If None, all columns
        will be selected.

    Returns:
        PixeltablePytorchDataset: The PyTorch dataset for the specified Ray node and CUDA device.
    """
    df = pxt_table.where(
        (col.node_id == node_id) & (col.cuda_device_id == cuda_device_id)
    ).select(select_exprs if select_exprs is not None else pxt_table.columns())

    return df.to_pytorch_dataset()
