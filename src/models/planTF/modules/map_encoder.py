import torch
import torch.nn as nn

from ..layers.embedding import PointsEncoder


class MapEncoder(nn.Module):
    def __init__(
        self,
        polygon_channel=6,
        dim=128,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.polygon_encoder = PointsEncoder(polygon_channel, dim)
        self.speed_limit_emb = nn.Sequential(
            nn.Linear(1, dim), nn.ReLU(), nn.Linear(dim, dim)
        )

        self.type_emb = nn.Embedding(3, dim)
        self.on_route_emb = nn.Embedding(2, dim)
        self.traffic_light_emb = nn.Embedding(4, dim)
        self.unknown_speed_emb = nn.Embedding(1, dim)

    def forward(self, data) -> torch.Tensor:
        polygon_center = data["map"]["polygon_center"]
        polygon_type = data["map"]["polygon_type"].long()
        polygon_on_route = data["map"]["polygon_on_route"].long()
        polygon_tl_status = data["map"]["polygon_tl_status"].long()
        polygon_has_speed_limit = data["map"]["polygon_has_speed_limit"]
        polygon_speed_limit = data["map"]["polygon_speed_limit"]
        point_position = data["map"]["point_position"]
        point_vector = data["map"]["point_vector"]
        point_orientation = data["map"]["point_orientation"]
        valid_mask = data["map"]["valid_mask"]

        polygon_feature = torch.cat(
            [
                point_position[:, :, 0] - polygon_center[..., None, :2],
                point_vector[:, :, 0],
                torch.stack(
                    [
                        point_orientation[:, :, 0].cos(),
                        point_orientation[:, :, 0].sin(),
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )

        bs, M, P, C = polygon_feature.shape
        valid_mask = valid_mask.view(bs * M, P)
        polygon_feature = polygon_feature.reshape(bs * M, P, C)

        x_polygon = self.polygon_encoder(polygon_feature, valid_mask).view(bs, M, -1)

        x_type = self.type_emb(polygon_type)
        x_on_route = self.on_route_emb(polygon_on_route)
        x_tl_status = self.traffic_light_emb(polygon_tl_status)
        x_speed_limit = torch.zeros(bs, M, self.dim, device=x_polygon.device)
        x_speed_limit[polygon_has_speed_limit] = self.speed_limit_emb(
            polygon_speed_limit[polygon_has_speed_limit].unsqueeze(-1)
        )
        x_speed_limit[~polygon_has_speed_limit] = self.unknown_speed_emb.weight

        x_polygon += x_type + x_on_route + x_tl_status + x_speed_limit

        return x_polygon
