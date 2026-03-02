use crate::types::FaceFeature;
use anyhow::{anyhow, Result};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

const VECTOR_DIMENSIONS: usize = 512;

pub struct FaceSearchIndex {
    index: Index,
    next_id: u64,
}

impl FaceSearchIndex {
    /// 创建新的人脸搜索索引，使用 USearch HNSW 算法
    ///
    /// # Returns
    ///
    /// * `Result<FaceSearchIndex>` - 成功时返回索引实例，失败时返回错误  Ok Err
    ///
    /// # Examples
    ///
    /// ```
    /// use face_rust::FaceSearchIndex;
    /// let mut index = FaceSearchIndex::new()?;
    /// ```
    pub fn new() -> Result<Self> {
        // 配置 USearch HNSW 选项
        let options = IndexOptions {
            dimensions: VECTOR_DIMENSIONS,
            metric: MetricKind::L2sq, //2026-1-6 wuxinyi 修改为L2sq
            quantization: ScalarKind::F32,
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 128,
            multi: false,
            ..Default::default()
        };
        let index = Index::new(&options).map_err(|e| anyhow!("options failed!: {}", e))?;
        index.reserve(1000).expect("Failed to reserve capacity.");
        Ok(Self {
            index,
            next_id: 0,
        })
    }

    // debug
    pub fn debug(&self, feature: &FaceFeature) -> Result<()> {
        if feature.vector.len() == 0 {
            return Err(anyhow!("feature vector null!"));
        }
        if feature.vector.len() != VECTOR_DIMENSIONS {
            return Err(anyhow!(
                "faeture vector dim not match!: expect {}, actual {}",
                VECTOR_DIMENSIONS,
                feature.vector.len()
            ));
        }
        Ok(())
    }

    /// 向索引中添加人脸特征（指定ID）
    ///
    /// # Arguments
    ///
    /// * `id` - 人脸的唯一标识符
    /// * `feature` - 人脸特征向量
    /// # Returns
    /// * `Result<()>` - 成功时返回 Ok(())，失败时返回错误
    pub fn add_id(&mut self, id: u64, feature: &FaceFeature) -> Result<()> {
        self.debug(feature)?;
        self.index
            .add(id, &feature.vector)
            .map_err(|e| anyhow!("USearch add failed!: {}", e))?;
        Ok(())
    }

    pub fn add_batch<T>(&mut self, features: &[T]) -> Result<Vec<u64>>
    where
        T: AsRef<FaceFeature>,
    {
        let start_id = self.next_id;
        let mut ids = Vec::with_capacity(features.len());

        // 批量添加
        for (i, feature) in features.iter().enumerate() {
            let id = start_id + i as u64;
            let feature_ref = feature.as_ref();

            self.index
                .add(id, &feature_ref.vector)
                .map_err(|e| anyhow!("添加第 {} 个特征失败: {}", i + 1, e))?;

            ids.push(id);
        }

        // 更新 next_id
        self.next_id = start_id + features.len() as u64;

        println!("批量添加成功: {} 个特征", ids.len());
        Ok(ids)
    }

    /// 向索引中添加人脸特征（自动分配ID）
    ///
    /// # Arguments
    ///
    /// * `feature` - 人脸特征向量
    ///
    /// # Returns
    ///
    /// * `Result<u64>` - 成功时返回分配的ID，失败时返回错误
    pub fn add_face(&mut self, feature: &FaceFeature) -> Result<u64> {
        let id = self.next_id;
        self.add_id(id, feature)?;
        self.next_id += 1;
        Ok(id)
    }

    /// 在索引中搜索与查询特征最相似的人脸
    /// 使用 USearch HNSW 算法进行高效近似最近邻搜索
    ///
    /// # Arguments
    ///
    /// * `query` - 查询的人脸特征
    /// * `k` - 返回的最相似结果数量
    ///
    /// # Returns
    ///
    /// * `Result<Vec<(u64, f32)>>` - 成功时返回 (ID, 相似度) 列表，按相似度降序排列
    pub fn search(&self, query: &FaceFeature, k: usize) -> Result<Vec<(u64, f32)>> {
        // 验证输入参数
        self.debug(query)?;
        let search_results = Index::search(&self.index, &query.vector, k)
            .map_err(|e| anyhow!("search failed!: {}", e))?;

        //2026-1-5 wuxinyi 返回原始数据，由业务层进行准确比较
        let results = search_results
            .keys
            .iter()
            .zip(search_results.distances.iter())
            .map(|(&key, &dist)| (key, dist))
            .collect();

        Ok(results)
    }

    /// 在索引中进行精确搜索（暴力搜索）
    ///
    /// # Arguments
    ///
    /// * `query` - 查询的人脸特征
    /// * `k` - 返回的最相似结果数量
    ///
    /// # Returns
    ///
    /// * `Result<Vec<(u64, f32)>>` - 成功时返回 (ID, 相似度) 列表
    pub fn exact_search(&self, query: &FaceFeature, k: usize) -> Result<Vec<(u64, f32)>> {
        // 验证输入参数
        self.debug(query)?;
        // 使用 VectorType trait 的 exact_search 方法
        let search_results = Index::exact_search(&self.index, &query.vector, k)
            .map_err(|e| anyhow!("exact_search failed: {},在{}{}", e, file!(), line!()))?;
        // 转换为相似度
        let mut results: Vec<(u64, f32)> = Vec::with_capacity(search_results.keys.len());
        for i in 0..search_results.keys.len() {
            let key = search_results.keys[i];
            let distance = search_results.distances[i];
            let similarity = 1.0 - distance;
            results.push((key, similarity));
        }
        // 按相似度排序（降序）
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        println!("USearch搜索成功，返回{}个结果", results.len());
        Ok(results)
    }

    /// 从索引中获取指定ID的人脸特征
    ///
    /// # Arguments
    ///
    /// * `id` - 要获取的人脸ID
    ///
    /// # Returns
    ///
    /// * `Result<FaceFeature>` - 成功时返回人脸特征，失败时返回错误
    pub fn get_face(&self, id: u64) -> Result<FaceFeature> {
        let mut buffer = vec![0.0f32; VECTOR_DIMENSIONS];
        Index::get(&self.index, id, &mut buffer)
            .map_err(|e| anyhow!("USearch 获取向量失败: {}", e))?;

        if buffer.len() != VECTOR_DIMENSIONS {
            return Err(anyhow!(
                "检索到的向量维度不匹配: 期望 {}, 实际 {}",
                VECTOR_DIMENSIONS,
                buffer.len()
            ));
        }
        Ok(FaceFeature::new_with_id(id, buffer))
    }

    /// 获取索引中存储的人脸数量
    pub fn len(&self) -> usize {
        self.index.size()
    }

    /// 更新指定id的人脸特征
    /// # Arguments
    /// * `id` - 要更新的人脸ID
    /// * `feature` - 新的人脸特征向量
    /// # Returns
    /// * `Result<()>` - 成功时返回 Ok(())，失败时返回错误
    pub fn update_face(&mut self, id: u64, feature: &FaceFeature) -> Result<()> {
        self.debug(feature)?;
        self.index
            .remove(id)
            .map_err(|e| anyhow!("remove failed: {}", e))?;
        self.index
            .add(id, &feature.vector)
            .map_err(|e| anyhow!("USearch 更新失败: {}", e))?;
        println!("USearch 人脸特征更新成功: ID={}", id);
        Ok(())
    }

    /// 保存索引到文件
    pub fn save(&self, path: &str) -> Result<()> {
        self.index
            .save(path)
            .map_err(|e| anyhow!("USearch 保存失败: {}", e))?;
        println!("USearch 索引已保存到: {}", path);
        Ok(())
    }

    /// 从文件加载索引
    pub fn load(path: &str) -> Result<Self> {
        // 首先创建一个新索引，然后从文件加载
        let options = IndexOptions {
            dimensions: VECTOR_DIMENSIONS,
            metric: MetricKind::L2sq,
            quantization: ScalarKind::F32,
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 128,
            multi: false,
            ..Default::default()
        };

        let index =
            Index::new(&options).map_err(|e| anyhow!("创建临时 USearch 索引失败: {}", e))?;

        index
            .load(path)
            .map_err(|e| anyhow!("USearch 加载失败: {}", e))?;

        println!("USearch 索引已从文件加载: {}", path);
        Ok(Self {
            index,
            next_id: 0,
        })
    }

    /// 获取下一个将要分配的ID
    pub fn next_id(&self) -> u64 {
        self.next_id
    }

    /// 清空索引中的所有数据
    pub fn clear(&mut self) -> Result<()> {
        self.index
            .reset()
            .map_err(|e| anyhow!("USearch 清空失败: {}", e))?;
        self.next_id = 0;
        Ok(())
    }

    /// 更改距离度量函数
    ///
    /// # Arguments
    ///
    /// * `metric` - 新的距离度量函数
    ///
    /// # Returns
    ///
    /// * `Result<()>` - 成功时返回 Ok(())，失败时返回错误
    pub fn change_metric(
        &mut self,
        metric: Box<dyn Fn(*const f32, *const f32) -> f32 + Send + Sync>,
    ) -> Result<()> {
        Index::change_metric(&mut self.index, metric);
        println!("USearch 距离度量已更新");
        Ok(())
    }
}
