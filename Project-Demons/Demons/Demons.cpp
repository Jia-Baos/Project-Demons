#include "Demons.h"

Demons::Demons(const demons_params_t& demons_params)
{
	this->demons_params = demons_params;

	std::cout << "Initalize the class, own..." << std::endl;
}

Demons::Demons()
{
	std::cout << "Initalize the class, pure..." << std::endl;
}

Demons::~Demons()
{
	std::cout << "Destory the class..." << std::endl;
}

void Demons::SingleScale(const cv::Mat& fixed_image, const cv::Mat& moved_image)
{
	cv::cvtColor(fixed_image, this->fixed_image_, cv::COLOR_BGR2GRAY);
	cv::cvtColor(moved_image, this->moved_image_, cv::COLOR_BGR2GRAY);

	cv::Mat flow_x = cv::Mat::zeros(this->fixed_image_.size(), CV_32FC1);
	cv::Mat flow_y = cv::Mat::zeros(this->fixed_image_.size(), CV_32FC1);
	float cc = DemonsSingle(this->fixed_image_, this->moved_image_,
		flow_x, flow_y,
		demons_params.niter_,
		demons_params.alpha_,
		demons_params.sigma_fluid_,
		demons_params.sigma_diffusion_);

	std::vector<cv::Mat> flow{ flow_x,flow_y };
	cv::merge(flow, this->flow_);

	MovePixels(this->moved_image_, this->moved_image_warpped_,
		flow_x, flow_y, cv::INTER_CUBIC);
	this->moved_image_warpped_.convertTo(this->moved_image_warpped_, CV_8UC1);

	cv::absdiff(this->fixed_image_, this->moved_image_warpped_, this->res_image_);
}

void Demons::MultiScale(const cv::Mat& fixed_image, const cv::Mat& moved_image)
{
	cv::cvtColor(fixed_image, this->fixed_image_, cv::COLOR_BGR2GRAY);
	cv::cvtColor(moved_image, this->moved_image_, cv::COLOR_BGR2GRAY);

	std::vector<cv::Mat> fixed_image_pyramid;
	std::vector<cv::Mat> moved_image_pyramid;
	GaussPyramid(this->fixed_image_, fixed_image_pyramid);
	GaussPyramid(this->moved_image_, moved_image_pyramid);

	cv::Mat flow_x = cv::Mat::zeros(this->fixed_image_.size(), CV_32FC1);
	cv::Mat flow_y = cv::Mat::zeros(this->fixed_image_.size(), CV_32FC1);

	DemonsMulti(fixed_image_pyramid, moved_image_pyramid,
		flow_x, flow_y,
		demons_params.niter_,
		demons_params.alpha_,
		demons_params.sigma_fluid_,
		demons_params.sigma_diffusion_);

	std::vector<cv::Mat> flow{ flow_x,flow_y };
	cv::merge(flow, this->flow_);

	MovePixels(this->moved_image_, this->moved_image_warpped_,
		flow_x, flow_y, cv::INTER_CUBIC);
	this->moved_image_warpped_.convertTo(this->moved_image_warpped_, CV_8UC1);

	cv::absdiff(this->fixed_image_, this->moved_image_warpped_, this->res_image_);
}

void Demons::DemonsRefinement(const cv::Mat& fixed_image, const cv::Mat& moved_image, cv::Mat& flow)
{
	cv::cvtColor(fixed_image, this->fixed_image_, cv::COLOR_BGR2GRAY);
	cv::cvtColor(moved_image, this->moved_image_, cv::COLOR_BGR2GRAY);

	cv::Mat moved_image_temp;
	std::vector<cv::Mat> flow_split;
	cv::split(flow, flow_split);
	MovePixels(this->moved_image_, moved_image_temp,
		flow_split[0], flow_split[1], cv::INTER_CUBIC);

	cv::Mat flow_x = cv::Mat::zeros(this->fixed_image_.size(), CV_32FC1);
	cv::Mat flow_y = cv::Mat::zeros(this->fixed_image_.size(), CV_32FC1);
	float cc = DemonsSingle(this->fixed_image_, moved_image_temp,
		flow_x,
		flow_y,
		demons_params.niter_,
		demons_params.alpha_,
		demons_params.sigma_fluid_,
		demons_params.sigma_diffusion_);

	flow_split[0] = flow_split[0] + flow_x;
	flow_split[1] = flow_split[1] + flow_y;
	cv::merge(flow_split, flow);
}

/************************************************************************/
/*    Demons Single                                                     */
/************************************************************************/

float DemonsSingle(const cv::Mat& S0,
	const cv::Mat& M0,
	cv::Mat& sx,
	cv::Mat& sy,
	const int niter,
	const float alpha,
	const float sigma_fluid,
	const  float sigma_diffusion)
{
	// ���ο�ͼ��͸���ͼ��ת��Ϊ�����;���
	cv::Mat S, M;
	S0.convertTo(S, CV_32FC1);
	M0.convertTo(M, CV_32FC1);
	cv::Mat M1 = M.clone();

	// ��ǰ������ƶ�
	float cc_min = FLT_MIN;

	// ���ڴ洢��ǰ���λ�Ƴ������Ͻ��и���
	cv::Mat sx_min, sy_min;

	// ��ʼ���ٶȳ�Ϊ0
	cv::Mat vx = cv::Mat::zeros(S.size(), CV_32FC1);
	cv::Mat vy = cv::Mat::zeros(S.size(), CV_32FC1);

	for (int i = 0; i < niter; i++)
	{
		// �ڴ˴��޸ļ���λ�Ƴ��ķ���
		cv::Mat ux, uy;
		ActiveDemonsForce(S, M1, ux, uy, alpha);
		//ThirionDemonsForce(S, M1, ux, uy, alpha);
		//SymmetricDemonsForce(S, M1, ux, uy, alpha);

		// ��˹�˲��Լ����λ�Ƴ�����ƽ��
		GaussianSmoothing(ux, ux, sigma_fluid);
		GaussianSmoothing(uy, uy, sigma_fluid);

		// ��λ�Ƴ��ۼ�
		vx = vx + 0.75 * ux;
		vy = vy + 0.75 * uy;

		// �ٴ����ø�˹�˲��Լ����λ�Ƴ�����ƽ��
		GaussianSmoothing(vx, vx, sigma_diffusion);
		GaussianSmoothing(vy, vy, sigma_diffusion);

		// ���ۼӵ�λ�Ƴ�ת��Ϊ΢��ͬ��ӳ��
		ExpField(vx, vy, sx, sy);

		cv::Mat mask;
		// �����ɫ��Ե��mask�������
		ComputeMask(sx, sy, mask);
		// �Ը���ͼ��M���������ز���
		MovePixels(M, M1, sx, sy, cv::INTER_CUBIC);
		// ����F��M1�����ƶ�
		float cc_curr = ComputeCCMask(S, M1, mask);

		if (cc_curr > cc_min)
		{
			// ������ϵ����ߣ������M1��������ϵ����λ�Ƴ�
			// ��Ҫע����ǣ���ʹ��ǰ�ĸ���ʹ�����ϵ�����ͣ�����Ҳ�������θ��µĽ���������Ǿܾ��˴θ��²�����λ�Ƴ�
			// ���ǿ������Ϊ�㷨�ڴ�ʱ����ֲ���ֵ�������ɷ��ϵ��Ǽ����������п��ܻ�ȡ���õĽ���ģ��ʲ�Ӧ�ڴ�ʱ��ֹ����
			std::cout << "epoch = " << i << "; cc = " << cc_min << std::endl;
			cc_min = cc_curr;
			sx_min = sx.clone();
			sy_min = sy.clone();
		}
	}

	// �õ���ǰ������΢��ͬ��ӳ��
	sx = sx_min.clone();
	sy = sy_min.clone();
	return cc_min;
}

/************************************************************************/
/*    Demons Multi                                                     */
/************************************************************************/

void DemonsMulti(const std::vector<cv::Mat>& fixed_image_pyramid,
	const std::vector<cv::Mat>& moved_image_pyramid,
	cv::Mat& sx,
	cv::Mat& sy,
	const int niter,
	const float alpha,
	const float sigma_fluid,
	const  float sigma_diffusion)
{
	const int layers = fixed_image_pyramid.size();
	std::vector<cv::Mat> moved_image_pyramid_tmp = moved_image_pyramid;

	// ���ڴ洢��ǰ���λ�Ƴ������Ͻ��и���
	cv::Mat sx_min, sy_min;

	for (int iter = 0; iter < layers; iter++)
	{
		std::cout << "curr layer: " << iter << std::endl;

		const cv::Mat fixed_image_curr = fixed_image_pyramid[iter];
		const cv::Mat moved_image_curr = moved_image_pyramid_tmp[iter];

		// ��ʼ���ٶȳ�Ϊ0
		cv::Mat vx = cv::Mat::zeros(fixed_image_curr.size(), CV_32FC1);
		cv::Mat vy = cv::Mat::zeros(fixed_image_curr.size(), CV_32FC1);

		// ���ο�ͼ��͸���ͼ��ת��Ϊ�����;���
		cv::Mat S, M;
		fixed_image_curr.convertTo(S, CV_32FC1);
		moved_image_curr.convertTo(M, CV_32FC1);
		cv::Mat M1 = M.clone();

		// ��ǰ������ƶ�
		float cc_min = FLT_MIN;

		for (int i = 0; i < niter; i++)
		{
			// �ڴ˴��޸ļ���λ�Ƴ��ķ���
			cv::Mat ux, uy;
			ActiveDemonsForce(S, M1, ux, uy, alpha);
			//ThirionDemonsForce(S, M1, ux, uy, alpha);
			//SymmetricDemonsForce(S, M1, ux, uy, alpha);

			// ��˹�˲��Լ����λ�Ƴ�����ƽ��
			GaussianSmoothing(ux, ux, sigma_fluid);
			GaussianSmoothing(uy, uy, sigma_fluid);

			// ��λ�Ƴ��ۼ�
			vx = vx + 0.75 * ux;
			vy = vy + 0.75 * uy;

			// �ٴ����ø�˹�˲��Լ����λ�Ƴ�����ƽ��
			GaussianSmoothing(vx, vx, sigma_diffusion);
			GaussianSmoothing(vy, vy, sigma_diffusion);

			// ���ۼӵ�λ�Ƴ�ת��Ϊ΢��ͬ��ӳ��
			ExpField(vx, vy, sx, sy);

			cv::Mat mask;
			// �����ɫ��Ե��mask�������
			ComputeMask(sx, sy, mask);
			// �Ը���ͼ��M���������ز���
			MovePixels(M, M1, sx, sy, cv::INTER_CUBIC);
			// ����F��M1�����ƶ�
			const float cc_curr = ComputeCCMask(S, M1, mask);

			if (cc_curr > cc_min)
			{
				// ������ϵ����ߣ������M1��������ϵ����λ�Ƴ�
				// ��Ҫע����ǣ���ʹ��ǰ�ĸ���ʹ�����ϵ�����ͣ�����Ҳ�������θ��µĽ���������Ǿܾ��˴θ��²�����λ�Ƴ�
				// ���ǿ������Ϊ�㷨�ڴ�ʱ����ֲ���ֵ�������ɷ��ϵ��Ǽ����������п��ܻ�ȡ���õĽ���ģ��ʲ�Ӧ�ڴ�ʱ��ֹ����
				std::cout << "epoch = " << i << "; cc = " << cc_min << std::endl;
				cc_min = cc_curr;
				sx_min = sx.clone();
				sy_min = sy.clone();
			}
		}

		// �õ���ǰ������΢��ͬ��ӳ�䣬��������ײ���Ҫ�����ϲ������Ա������һ�ε���
		sx = sx_min.clone();
		sy = sy_min.clone();

		if (iter < layers - 1)
		{
			cv::pyrUp(sx, sx, fixed_image_pyramid[iter + 1].size());
			cv::pyrUp(sy, sy, fixed_image_pyramid[iter + 1].size());

			// ���ݵ�ǰ�������������΢��ͬ��ӳ�����һ��ĸ���ͼ��ͼ�����wrap
			MovePixels(moved_image_pyramid[iter + 1], moved_image_pyramid_tmp[iter + 1], sx, sy, cv::INTER_CUBIC);
		}
	}
}

/************************************************************************/
/*    Driving force                                                     */
/************************************************************************/

void ComputeGradient(const cv::Mat& src, cv::Mat& Fx, cv::Mat& Fy)
{
	cv::Mat src_board;
	cv::copyMakeBorder(src, src_board, 1, 1, 1, 1, cv::BORDER_CONSTANT);
	Fx = cv::Mat::zeros(src.size(), CV_32FC1);
	Fy = cv::Mat::zeros(src.size(), CV_32FC1);

	for (int i = 0; i < src.rows; i++)
	{
		float* p_Fx = Fx.ptr<float>(i);
		float* p_Fy = Fy.ptr<float>(i);

		for (int j = 0; j < src.cols; j++)
		{
			// ˮƽ������ݶ�
			p_Fx[j] = (src_board.ptr<float>(i + 1)[j + 2] - src_board.ptr<float>(i + 1)[j]) / 2.0;
			// ��ֱ������ݶ�
			p_Fy[j] = (src_board.ptr<float>(i + 2)[j + 1] - src_board.ptr<float>(i)[j + 1]) / 2.0;
		}
	}
}

void ThirionDemonsForce(const cv::Mat& S, const cv::Mat& M,
	cv::Mat& Tx, cv::Mat& Ty, const  float alpha)
{
	// �󸡶�ͼ��M��ο�ͼ��S�ĻҶȲDiff
	cv::Mat Diff = S - M;
	Tx = cv::Mat::zeros(S.size(), CV_32FC1);
	Ty = cv::Mat::zeros(S.size(), CV_32FC1);

	// ��ο�ͼ��S���ݶ�
	cv::Mat Sx, Sy;
	ComputeGradient(S, Sx, Sy);

	// �󸡶�ͼ��M���ݶ�
	cv::Mat Mx, My;
	ComputeGradient(M, Mx, My);

	for (int i = 0; i < S.rows; i++)
	{
		// �ο�ͼ��S�ݶȵ�ָ��
		float* p_sx = Sx.ptr<float>(i);
		float* p_sy = Sy.ptr<float>(i);
		// ����ͼ��M�ݶȵ�ָ��
		float* p_mx = Mx.ptr<float>(i);
		float* p_my = My.ptr<float>(i);
		// λ�Ƴ�T��ָ��
		float* p_tx = Tx.ptr<float>(i);
		float* p_ty = Ty.ptr<float>(i);
		// �ҶȲDiff��ָ��
		float* p_diff = Diff.ptr<float>(i);

		for (int j = 0; j < S.cols; j++)
		{
			// ԭʼDemons��ֻ���ǲο�ͼ��S�γɵ�������
			float a = p_sx[j] * p_sx[j] + p_sy[j] * p_sy[j] + alpha * alpha * p_diff[j] * p_diff[j];

			// �Է�ĸ���нضϴ���
			if (a < -0.0000001 || a > 0.0000001)
			{
				p_tx[j] = p_diff[j] * (p_sx[j] / a);
				p_ty[j] = p_diff[j] * (p_sy[j] / a);
			}
		}
	}
}

void ActiveDemonsForce(const cv::Mat& S, const cv::Mat& M,
	cv::Mat& Tx, cv::Mat& Ty, const  float alpha)
{
	// �󸡶�ͼ��M��ο�ͼ��S�ĻҶȲDiff
	cv::Mat Diff = S - M;
	Tx = cv::Mat::zeros(S.size(), CV_32FC1);
	Ty = cv::Mat::zeros(S.size(), CV_32FC1);

	// ��ο�ͼ��S���ݶ�
	cv::Mat Sx, Sy;
	ComputeGradient(S, Sx, Sy);

	// �󸡶�ͼ��M���ݶ�
	cv::Mat Mx, My;
	ComputeGradient(M, Mx, My);

	for (int i = 0; i < S.rows; i++)
	{
		// �ο�ͼ��S�ݶȵ�ָ��
		float* p_sx = Sx.ptr<float>(i);
		float* p_sy = Sy.ptr<float>(i);
		// ����ͼ��M�ݶȵ�ָ��
		float* p_mx = Mx.ptr<float>(i);
		float* p_my = My.ptr<float>(i);
		// λ�Ƴ�T��ָ��
		float* p_tx = Tx.ptr<float>(i);
		float* p_ty = Ty.ptr<float>(i);
		// �ҶȲDiff��ָ��
		float* p_diff = Diff.ptr<float>(i);

		for (int j = 0; j < S.cols; j++)
		{
			float a1 = p_sx[j] * p_sx[j] + p_sy[j] * p_sy[j] + alpha * alpha * p_diff[j] * p_diff[j];
			float a2 = p_mx[j] * p_mx[j] + p_my[j] * p_my[j] + alpha * alpha * p_diff[j] * p_diff[j];

			// �Է�ĸ���нضϴ���
			if ((a1 < -0.0000001 || a1 > 0.0000001) && (a2 < -0.0000001 || a2 > 0.0000001))
			{
				p_tx[j] = p_diff[j] * (p_sx[j] / a1 + p_mx[j] / a2);
				p_ty[j] = p_diff[j] * (p_sy[j] / a1 + p_my[j] / a2);
			}
		}
	}
}

void SymmetricDemonsForce(const cv::Mat& S, const cv::Mat& M,
	cv::Mat& Tx, cv::Mat& Ty, const  float alpha)
{
	// �󸡶�ͼ��M��ο�ͼ��S�ĻҶȲDiff
	cv::Mat diff = S - M;
	Tx = cv::Mat::zeros(S.size(), CV_32FC1);
	Ty = cv::Mat::zeros(S.size(), CV_32FC1);

	// ��ο�ͼ��S���ݶ�
	cv::Mat Sx, Sy;
	ComputeGradient(S, Sx, Sy);

	// �󸡶�ͼ��M���ݶ�
	cv::Mat Mx, My;
	ComputeGradient(M, Mx, My);

	for (int i = 0; i < S.rows; i++)
	{
		// �ο�ͼ��S�ݶȵ�ָ��
		float* p_sx = Sx.ptr<float>(i);
		float* p_sy = Sy.ptr<float>(i);
		// ����ͼ��M�ݶȵ�ָ��
		float* p_mx = Mx.ptr<float>(i);
		float* p_my = My.ptr<float>(i);
		// λ�Ƴ�T��ָ��
		float* p_tx = Tx.ptr<float>(i);
		float* p_ty = Ty.ptr<float>(i);
		// �ҶȲDiff��ָ��
		float* p_diff = diff.ptr<float>(i);

		for (int j = 0; j < S.cols; j++)
		{
			float ax = (p_sx[j] + p_mx[j]) * (p_sx[j] + p_mx[j]) + (p_sy[j] + p_my[j]) * (p_sy[j] + p_my[j]) + 4 * alpha * alpha * p_diff[j] * p_diff[j];
			float ay = (p_sx[j] + p_mx[j]) * (p_sx[j] + p_mx[j]) + (p_sy[j] + p_my[j]) * (p_sy[j] + p_my[j]) + 4 * alpha * alpha * p_diff[j] * p_diff[j];

			//float ax = (p_sx[j] + p_mx[j]) * (p_sx[j] + p_mx[j]) + (p_sy[j] + p_my[j]) * (p_sy[j] + p_my[j]) + alpha * alpha * p_diff[j] * p_diff[j];
			//float ay = (p_sx[j] + p_mx[j]) * (p_sx[j] + p_mx[j]) + (p_sy[j] + p_my[j]) * (p_sy[j] + p_my[j]) + alpha * alpha * p_diff[j] * p_diff[j];

			// �Է�ĸ���нضϴ���
			if ((ax < -0.0000001 || ax > 0.0000001) && (ay < -0.0000001 || ay > 0.0000001))
			{
				p_tx[j] = 2 * p_diff[j] * (p_sx[j] + p_mx[j]) / ax;
				p_ty[j] = 2 * p_diff[j] * (p_sy[j] + p_my[j]) / ay;

				// p_tx[j] = p_diff[j] * (p_sx[j] + p_mx[j]) / ax;
				// p_ty[j] = p_diff[j] * (p_sy[j] + p_my[j]) / ay;
			}
		}
	}
}

/************************************************************************/
/*    Smoothing                                                         */
/************************************************************************/

void GaussianSmoothing(const cv::Mat& src, cv::Mat& dst, const float sigma)
{
	// ����ȡ��
	int radius = static_cast<int>(std::ceil(3 * sigma));
	// ����radiusΪ�滹��ż��ksizeʼ��Ϊ��
	int ksize = 2 * radius + 1;

	cv::GaussianBlur(src, dst, cv::Size(ksize, ksize), sigma);
}

void GaussPyramid(const cv::Mat& src,
	std::vector<cv::Mat>& gauss_pyramid, const int min_width)
{
	const float py_ratio = 0.5;
	const int max_col_row = src.cols > src.rows ? src.cols : src.rows;
	const int layers = std::log(static_cast<float>(min_width) / max_col_row)
		/ std::log(py_ratio);

	// ����ͼ�������
	cv::Mat current_img = src;
	gauss_pyramid.emplace_back(current_img);
	for (int i = 0; i < layers - 1; i++)
	{
		cv::Mat temp_img;
		cv::pyrDown(current_img, temp_img, cv::Size(current_img.cols / 2, current_img.rows / 2));
		gauss_pyramid.emplace_back(temp_img);
		current_img = temp_img;
	}
	std::reverse(gauss_pyramid.begin(), gauss_pyramid.end());
}

/************************************************************************/
/*    Metrics                                                           */
/************************************************************************/

void ComputeMask(const cv::Mat& Tx, const cv::Mat& Ty, cv::Mat& mask)
{
	mask = cv::Mat::zeros(Tx.size(), CV_8UC1);

	for (int i = 0; i < Tx.rows; i++)
	{
		const float* p_Tx = Tx.ptr<float>(i);
		const float* p_Ty = Ty.ptr<float>(i);
		uchar* p_mask = mask.ptr<uchar>(i);

		for (int j = 0; j < Tx.cols; j++)
		{
			int x = static_cast<int>(j + p_Tx[j]);
			int y = static_cast<int>(i + p_Ty[j]);

			if (x > 0 && x < Tx.cols && y > 0 && y < Tx.rows)
			{
				p_mask[j] = 255;
			}
		}
	}
}

double ComputeCCMask(const cv::Mat& S, const cv::Mat& Mi, const cv::Mat& Mask)
{
	float sum1 = 0.0;
	float sum2 = 0.0;
	float sum3 = 0.0;

	for (int i = 0; i < S.rows; i++)
	{
		const float* p_S = S.ptr<float>(i);
		const float* p_Mi = Mi.ptr<float>(i);
		for (int j = 0; j < S.cols; j++)
		{
			// ӳ��󳬱߽�ĵ㲻����ͳ��
			if (Mask.ptr<uchar>(i)[j])
			{
				float S_value = p_S[j];
				float Mi_value = p_Mi[j];
				sum1 += S_value * Mi_value;
				sum2 += S_value * S_value;
				sum3 += Mi_value * Mi_value;
			}
		}
	}

	// ��һ��
	const float result = sum1 / std::sqrt(sum2 * sum3);
	return result;
}

/************************************************************************/
/*    Remap and Exp Composite                                           */
/************************************************************************/

void MovePixels(const cv::Mat& src, cv::Mat& dst,
	const cv::Mat& Tx, const cv::Mat& Ty,
	const int interpolation)
{
	cv::Mat Tx_map(src.size(), CV_32FC1, 0.0);
	cv::Mat Ty_map(src.size(), CV_32FC1, 0.0);

	for (int i = 0; i < src.rows; i++)
	{
		float* p_Tx_map = Tx_map.ptr<float>(i);
		float* p_Ty_map = Ty_map.ptr<float>(i);
		for (int j = 0; j < src.cols; j++)
		{
			p_Tx_map[j] = j + Tx.ptr<float>(i)[j];
			p_Ty_map[j] = i + Ty.ptr<float>(i)[j];
		}
	}

	cv::remap(src, dst, Tx_map, Ty_map, interpolation);
}

void ExpComposite(cv::Mat& vx, cv::Mat& vy)
{
	// ��������ʵ��
	// ����x��y�����λ�Ƴ��ֱ�ΪUx��Uy
	// ʹ��Ux��Uy�ֱ������������������ز����Ĳ����õ�Ux'��Uy'��Ȼ���ټ���Ux+Ux'��Uy+Uy'��������Ǹ�������

	cv::Mat bxp, byp;
	MovePixels(vx, bxp, vx, vy, cv::INTER_CUBIC);
	MovePixels(vy, byp, vx, vy, cv::INTER_CUBIC);

	// ��������ı��ʾ��ǵݹ�˷�����
	vx = vx + bxp;
	vy = vy + byp;
}

void ExpField(const cv::Mat& vx, const cv::Mat& vy,
	cv::Mat& vx_out, cv::Mat& vy_out)
{
	// �����ж�Ӧλ�õ�Ԫ�����
	cv::Mat normv2 = vx.mul(vx) + vy.mul(vy);

	// �����ֵ����Сֵ
	double minv, maxv;
	cv::Point pt_min, pt_max;
	cv::minMaxLoc(normv2, &minv, &maxv, &pt_min, &pt_max);

	float m = std::sqrt(maxv);
	float n = std::ceil(std::log2(m / 0.5));
	n = n > 0.0 ? n : 0.0;

	float a = std::pow(2.0, -n);

	// ���ţ�ͨ�����½��ƿ��Ը��õ���߾���
	vx_out = vx * a;
	vy_out = vy * a;

	// n�θ������㣬���������ǵݹ�˷�����
	for (int i = 0; i < static_cast<int>(n); i++)
	{
		ExpComposite(vx_out, vy_out);
	}
}
