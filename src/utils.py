import os
import json
import random
import numpy as np
import torch, torchvision
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import Dataset, DataLoader
from data.algorithms.astar import AStarPlanner


class AD4NSDataset(Dataset):

    def __init__(self, root="./data/", version="mini", detector="FCOS3D", planner="AStar", device="cuda:0"):
        # define the supported dataset, detection models and planning algorithms
        self.supported_versions = ["mini"] + ["trainval" + "{:0>2d}".format(i + 1) for i in range(10)] + ["trainval"]
        self.supported_detectors = ["FCOS3D"]
        self.supported_planners = ["AStar"]
        # check if the args are supported
        self._supported_check(version, detector, planner)
        # check if the needed path are given
        self._path_check(root, version, detector, planner)
        # get the length of the dataset
        self._get_len()
        # load the detection model as detector
        self._load_model(device)
        # load the planning algorithm as planner
        self._load_alg()

    def __len__(self):
        return np.sum(self.dataset_len)

    def __getitem__(self, idx, seed=None):
        if idx == "detector":
            return self.detector
        elif idx == "planner":
            return self.planner
        elif idx == "random":
            if seed:
                random.seed(seed)
            idx = random.choice(range(len(self)))

        if idx < 0:
            raise IndexError("The index cannot be negative")

        presum = 0
        for di, dl in enumerate(self.dataset_len):
            if idx < presum + dl:  # found the right dataset
                # get the relative sample data path
                dp = self.dataset_path[di]
                idx -= presum
                fp = str(os.path.join(dp, sorted(os.listdir(dp))[idx]))
                if not os.path.exists(fp):
                    raise FileNotFoundError("The sample data path \' " + fp + " \' is not found")
                # load the sample data files and return it with idx
                img, dif, dof, pif, pof = self._load_item(fp)
                return idx + presum, img, dif, dof, pif, pof
            presum += dl

        raise IndexError("The index is over the max limit {}".format(presum - 1))

    def _supported_check(self, version, detector, planner):
        if version not in self.supported_versions:
            raise NotImplementedError("The supported versions are {}".format(self.supported_versions))
        if detector not in self.supported_detectors:
            raise NotImplementedError("The supported detectors are {}".format(self.supported_detectors))
        if planner not in self.supported_planners:
            raise NotImplementedError("The supported planners are {}".format(self.supported_planners))

    def _path_check(self, root, version, detector, planner):
        # data path
        if version == "trainval":  # the whole trainvalx dataset, x = [01,02,..,09,10]
            self.dataset_path = []
            for x in ["0" + str(i) for i in range(1, 10)] + ["10"]:
                self.dataset_path.append(os.path.join(root, "nuscenes-" + version + x))
            for dp in self.dataset_path:
                if not os.path.exists(dp):
                    raise FileNotFoundError("The dataset path \' " + dp + " \' is not found")
        else:
            self.dataset_path = [os.path.join(root, "nuscenes-" + version)]
            if not os.path.exists(self.dataset_path[0]):
                raise FileNotFoundError("The dataset path \' " + self.dataset_path[0] + " \' is not found")
        # model path
        self.model_path = os.path.join(root, "models", detector + ".pth")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("The model path \' " + self.model_path + " \' is not found")
        # algorithm path
        self.alg_path = os.path.join(root, "algorithms", planner.lower() + ".py")
        if not os.path.exists(self.alg_path):
            raise FileNotFoundError("The algorithm path \' " + self.alg_path + " \' is not found")
        self.alg_name = planner

    def _get_len(self):
        self.dataset_len = []
        for dp in self.dataset_path:
            self.dataset_len.append(len(os.listdir(dp)))

    def _load_model(self, device):
        self.detector = torch.load(self.model_path)
        if "cuda" in device:
            if not torch.cuda.is_available():
                device = "cpu"
        self.detector.to(device)

    def _load_alg(self):
        if self.alg_name == "AStar":
            self.planner = AStarPlanner()
        else:
            pass

    def _load_item(self, fp):
        # load detection input image
        # shape = HxWxC, format: RGB, H = 900, W = 1600, dtype: np.uint8
        img_path = os.path.join(fp, "detection input.jpg")
        if not os.path.exists(img_path):
            raise FileNotFoundError("The detection input.jpg is not found in " + fp)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # load detection input info
        # dif is a dict with keys = {"name", "file_name", "image_id", "scene_id", "width", "height", "ego_pos", "cam_intrinsic"}
        # note that:
        #   dif["ego_pos"] is a list [x,y,z] representing the ego vehicle's position in camera coordinate
        #   dif["cam_intrinsic"] is a 3x3 matrix representing the transformation from camera coordinate to image coordinate
        dif_path = os.path.join(fp, "detection input info.json")
        if not os.path.exists(dif_path):
            raise FileNotFoundError("The detection input info.json in not found in " + fp)
        with open(dif_path, 'r') as f:
            dif = json.load(f)["images"][0]
        # load detection output info
        # dof is a dict with keys = {"detection model", "gt_bboxes", "pr_bboxes"}
        # note that:
        #   dof["xx_bboxes"] is a list with shape=(N,8,3), where N is the number of the obstacles in this image,
        #   and each obstacle is located with its bounding box(8 corners' positions in camera coordinate)
        #   dof["gt_bboxes"] stores the ground-truth bounding boxes of each obstacle
        #   dof["pr_bboxes"] stores the predicted bounding boxes of each obstacle
        dof_path = os.path.join(fp, "detection output info.json")
        if not os.path.exists(dof_path):
            raise FileNotFoundError("The detection output info.json in not found in " + fp)
        with open(dof_path, 'r') as f:
            dof = json.load(f)
        # load planning input info
        # pif is a dict with keys = {"start", "goal", "gt_pts", "pr_pts", "boundary"}
        # note that:
        #   pif["start"] and pif["goal"] represent start point and goal point in 2D planning graph
        #   pif["xx_pts"] is a list of dict {"ox": [..], "oy": [..]}, each of which represents one obstacle's boundary points in 2D planning graph
        #   pif["boundary"] is a dict {"ox": [..], "oy": [..]}, representing the global boundary points of the 2D planning graph
        pif_path = os.path.join(fp, "planning input info.json")
        if not os.path.exists(pif_path):
            raise FileNotFoundError("The planning input info.json in not found in " + fp)
        with open(pif_path, 'r') as f:
            pif = json.load(f)
        # load planning output info
        # pof is a dict with keys = {"gt", "pr"}, and each element is also a dict {"pathx":[..], "pathy":[..], "searchx":[..], "searchy":[..]}
        # note that
        #   pof["gt"] is the planning result in ground-truth planning graph, while pof["pr"] is the one in predicted planning graph
        #   pof["xx"]["pathx"], pof["xx"]["pathy"] represents the optimized planning path from start point to goal point
        #   pof["xx"]["searchx"], pof["xx"]["searchy"] represents the visited points during the searching process
        pof_path = os.path.join(fp, "planning output info.json")
        if not os.path.exists(pof_path):
            raise FileNotFoundError("The planning output info.json in not found in " + fp)
        with open(pof_path, 'r') as f:
            pof = json.load(f)

        return img, dif, dof, pif, pof

    def render_detection_input(self, idx, seed=None):
        _, img, _, _, _, _ = self.__getitem__(idx, seed)
        fig = plt.figure(figsize=(9,16))
        ax = fig.add_subplot(111)
        ax.set_title("Detection Input Image")
        ax.axis("off")
        ax.imshow(img)

    def render_detection_output(self, idx, seed=None, save_path=None):
        _, img, dif, dof, _, _ = self.__getitem__(idx, seed)
        gt_img = img.copy()
        pr_img = img.copy()
        cam_intrinsic = dif["cam_intrinsic"]

        # init axes.
        _, axes = plt.subplots(1, 2, figsize=(14,18))
        # set title
        axes[0].set_title("Detection Output Ground-Truth")
        axes[1].set_title("Detection Output Predicted")
        # show image.
        axes[0].imshow(gt_img)
        axes[1].imshow(pr_img)
        # show boxes.
        colors = ['red', 'skyblue', 'cyan', 'orange', 'green', 'yellow', 'tomato', 'pink']
        for i, key in [(0, "gt"), (1, "pr")]:
            # draw every bounding box
            for box in dof[key + "_bboxes"]:  # box.shape=(8,3)
                # randomly choose a color to draw
                color = random.choice(colors)
                # sort corners to let the first 4 corners be the front, while the last 4 corners be the rear
                corners = sorted(box, key=lambda c: -c[2])
                # transfer to image coordinate
                corners = self._view_points(np.array(corners).T,  # transpose to shape=(3,8)
                                            np.array(cam_intrinsic), normalize=True)[:2, :].T  # transpose back to shape=(8,3)
                # sort the front and rear to let the direction be clockwise
                corners = corners.tolist()
                corners[:4] = sorted(corners[:4], key=lambda c: -c[1])
                corners[4:] = sorted(corners[4:], key=lambda c: -c[1])
                corners[:2] = sorted(corners[:2], key=lambda c: c[0])
                corners[4:6] = sorted(corners[4:6], key=lambda c: c[0])
                corners[2:4] = sorted(corners[2:4], key=lambda c: -c[0])
                corners[6:] = sorted(corners[6:], key=lambda c: -c[0])
                corners = np.array(corners)
                # draw the sides
                for j in range(4):
                    axes[i].plot([corners[j][0], corners[j + 4][0]],
                                 [corners[j][1], corners[j + 4][1]], alpha=0.6, color=color, linewidth=1)
                # draw front and rear rectangles(3d)/lines(2d)
                self._draw_rect(axes[i], corners[:4], color)
                self._draw_rect(axes[i], corners[4:], color)
                # draw line indicating the front
                center_bottom_forward = np.mean(corners[2:4], axis=0)
                center_bottom = np.mean(corners[[2, 3, 7, 6]], axis=0)
                axes[i].plot([center_bottom[0], center_bottom_forward[0]],
                            [center_bottom[1], center_bottom_forward[1]], alpha=0.6, color=color, linewidth=1)
            # limit visible area
            axes[i].set_xlim(0, img.shape[1])
            axes[i].set_ylim(img.shape[0], 0)
            # hide axis
            axes[i].axis("off")
        # save figure
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, "detection output.jpg"), dpi=100, bbox_inches='tight', pad_inches=0)

    def render_planning_input(self, idx, seed=None, save_path=None):
        _, _, _, _, pif, _ = self.__getitem__(idx, seed)
        start, goal, boundary = pif["start"], pif["goal"], pif["boundary"]

        # init axes
        _, axes = plt.subplots(1, 2, figsize=(10, 6))
        axes[0].axis("equal")
        axes[1].axis("equal")
        # set title
        axes[0].set_title("Planning Input Ground-Truth")
        axes[1].set_title("Planning Input Predicted")
        # draw planning graph
        for i, key in [(0, "gt"), (1, "pr")]:
            self._draw_planning_graph(axes[i], start, goal, pif[key + "_pts"], boundary)
        # save figure
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, "planning input.jpg"), dpi=100, bbox_inches='tight', pad_inches=0)

    def render_planning_output(self, idx, seed=None, render_process=False, save_path=None, fps=30):
        _, _, _, _, pif, pof = self.__getitem__(idx, seed)
        start, goal, boundary = pif["start"], pif["goal"], pif["boundary"]

        # init axes
        _, axes = plt.subplots(1, 2, figsize=(10, 6))
        axes[0].axis("equal")
        axes[1].axis("equal")
        # set title
        axes[0].set_title("Planning Output Ground-Truth")
        axes[1].set_title("Planning Output Predicted")
        # draw graph
        for i, key in [(0, "gt"), (1, "pr")]:
            pathx, pathy, searchx, searchy = pof[key]["pathx"], pof[key]["pathy"], pof[key]["searchx"], pof[key]["searchy"]
            # draw planning graph
            self._draw_planning_graph(axes[i], start, goal, pif[key + "_pts"], boundary)
            # draw planned path
            axes[i].plot(pathx, pathy, "--r")
        # save figure
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, "planning output.jpg"), dpi=100, bbox_inches='tight', pad_inches=0)

        # draw planning process
        if render_process:
            for i, key in [(0, "gt"), (1, "pr")]:
                fig = plt.figure(figsize=(16, 9))
                ax = fig.add_subplot(111)
                pathx, pathy, searchx, searchy = pof[key]["pathx"], pof[key]["pathy"], pof[key]["searchx"], pof[key]["searchy"]
                # draw planning graph
                self._draw_planning_graph(ax, start, goal, pif[key + "_pts"], boundary)
                # draw search process
                artists = []
                for j in range(len(searchx)):  # searching process
                    frame = []
                    frame += ax.plot(searchx[0:j + 1], searchy[0:j + 1], "xc")
                    artists.append(frame)
                # draw planned path
                for _ in range(fps):  # keep planned path for 1 second in the process
                    frame = []
                    frame += ax.plot(searchx, searchy, "xc")
                    frame += ax.plot(pathx, pathy, "-r")
                    artists.append(frame)
                # write search process
                ani = animation.ArtistAnimation(fig=plt.gcf(), artists=artists, repeat=False, interval=10)
                gif_file = "planning output process " + ("ground-truth" if id == "gt" else "predicted") + ".gif"
                ani.save(os.path.join(save_path, gif_file), writer='pillow', fps=fps)

    def _draw_rect(self, ax, selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            ax.plot([prev[0], corner[0]], [prev[1], corner[1]], alpha=0.6, linewidth=1, color=color)
            prev = corner

    def _view_points(self, points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
        """
        This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
        orthographic projections. It first applies the dot product between the points and the view. By convention,
        the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
        normalization along the third dimension.
        For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
        For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
        For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
         all zeros) and normalize=False
        :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
        :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
            The projection should be such that the corners are projected onto the first 2 axis.
        :param normalize: Whether to normalize the remaining coordinate (along the third axis).
        :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
        """

        assert view.shape[0] <= 4
        assert view.shape[1] <= 4
        assert points.shape[0] == 3

        viewpad = np.eye(4)
        viewpad[:view.shape[0], :view.shape[1]] = view

        nbr_points = points.shape[1]

        # Do operation in homogenous coordinates.
        points = np.concatenate((points, np.ones((1, nbr_points))))
        points = np.dot(viewpad, points)
        points = points[:3, :]

        if normalize:
            points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

        return points

    def _draw_planning_graph(self, ax, start, goal, draw_boxes, boundary):
        # 1. draw start point(ego vehicle's position) with shape='^', color=blue
        ax.scatter(start[0], start[1], c='steelblue', marker='^', s=70)
        # 2. draw goal point(the goal position) with shape='*', color=red
        ax.scatter(goal[0], goal[1], c='r', marker='*', s=70)
        # 3. draw obstacles with shape='x', color=random
        for db in draw_boxes:
            ax.scatter(db["ox"], db["oy"], marker='x', s=20)
        # 4. draw boundary with shape='+', color=black
        ax.scatter(boundary["ox"], boundary["oy"], marker='+', s=20, c='k')


class AD4NSDataLoader(DataLoader):

    def __init__(self, dataset: AD4NSDataset,
                 batch_size=1, shuffle=False, num_workers=0):
        super().__init__(dataset, batch_size, shuffle,
                         collate_fn=self._collator,
                         num_workers=num_workers)

    def _collator(self, batch):
        idxs = [x[0] for x in batch]
        imgs = np.stack([x[1] for x in batch])
        difs = [x[2] for x in batch]
        dofs = [x[3] for x in batch]
        pifs = [x[4] for x in batch]
        pofs = [x[5] for x in batch]
        return idxs, imgs, difs, dofs, pifs, pofs

    def render(self, key, idx, render_process=False, save_path=None, fps=30, seed=None):
        if key in ["detection input", "di"]:
            return self.dataset.render_detection_input(idx, seed)
        if key in ["detection output", "do"]:
            return self.dataset.render_detection_output(idx, seed, save_path)
        if key in ["planning input", "pi"]:
            return self.dataset.render_planning_input(idx, seed, save_path)
        if key in ["planning output", "po"]:
            return self.dataset.render_planning_output(idx, seed, render_process, save_path, fps)

        raise KeyError("The supported render keys are {}".format(
            ["detection input", "di"] + ["detection output", "do"] +
            ["planning input", "pi"] + ["planning output", "po"]
        ))

