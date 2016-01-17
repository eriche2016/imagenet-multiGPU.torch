require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'

-- dataLoader类
local dataset = torch.class('dataLoader')

-- 关于dataLoader类的参数检查函数， 以确定dataLoader类被正确使用
local initcheck = argcheck{
   pack=true,
   -- 帮助信息
   help=[[
     A dataset class for images in a flat folder structure (folder-name is class-name).
     Optimized for extremely large datasets (upwards of 14 million images).
     Tested only on Linux (as it uses command-line linux utilities to scale up)
]],
   -- 检查路径，注意paths是一个table of paths
   {check=function(paths)  
       local out = true;
       for k,v in ipairs(paths) do
          if type(v) ~= 'string' then
             print('paths can only be of string input');
             -- 一旦发现v不是字符串， 即设定out为false
             out = false
          end
       end
       return out
   end,
    name="paths",
    type="table",
    help="Multiple paths of directories with images"},
    -- 检查对采样图片的size
   {name="sampleSize",
    type="table",
    help="a consistent sample size to resize the images"},
   -- 训练样本和测试样本的划分比例
   {name="split",
    type="number",
    help="Percentage of split to go to Training"
   },

   {name="samplingMode",
    type="string",
    help="Sampling mode: random | balanced ",
    default = "balanced"},

   {name="verbose",
    type="boolean",
    help="Verbose mode during initialization",
    default = false},

   {name="loadSize",
    type="table",
    help="a size to load the images to, initially",
    opt = true},
   -- 强制进行类标签和类索引的映射关系
   {name="forceClasses",
    type="table",
    help="If you want this loader to map certain classes to certain indices, "
       .. "pass a classes table that has {classname : classindex} pairs."
       .. " For example: {3 : 'dog', 5 : 'cat'}"
       .. "This function is very useful when you want two loaders to have the same "
    .. "class indices (trainLoader/testLoader for example)",
    opt = true},

   {name="sampleHookTrain",
    type="function",
    help="applied to sample during training(ex: for lighting jitter). "
       .. "It takes the image path as input",
    opt = true},

   {name="sampleHookTest",
    type="function",
    help="applied to sample during testing",
    opt = true},
}

function dataset:__init(...)
   -- argcheck
   local args =  initcheck(...)
   print(args)
   -- 进行相关参数赋值
   for k,v in pairs(args) do self[k] = v end
   -- self = {} 里面有相关参数的设置    
   if not self.loadSize then self.loadSize = self.sampleSize; end

   if not self.sampleHookTrain then self.sampleHookTrain = self.defaultSampleHook end
   if not self.sampleHookTest then self.sampleHookTest = self.defaultSampleHook end

   -- find class names
   -- 类名，dataset自己具有的变量域
   self.classes = {}
   --相关类样本文件存放的地址
   local classPaths = {}
   -- 是否强制制定类标签和类索引的映射关系。 例如制定的索引3对应dog类等。 
   if self.forceClasses then
      for k,v in pairs(self.forceClasses) do
         self.classes[k] = v
         -- 相关类图片存放的的路径
         classPaths[k] = {}
      end
   end
   
   local function tableFind(t, o) for k,v in pairs(t) do if v == o then return k end end end
   
   -- loop over each paths folder, get list of unique class names,
   -- also store the directory paths per class
   -- for each class,
   for k,path in ipairs(self.paths) do
      local dirs = dir.getdirectories(path);
      for k,dirpath in ipairs(dirs) do
         local class = paths.basename(dirpath)
         local idx = tableFind(self.classes, class)
         if not idx then
            table.insert(self.classes, class)
            idx = #self.classes
            classPaths[idx] = {}
         end
         if not tableFind(classPaths[idx], dirpath) then
            table.insert(classPaths[idx], dirpath);
         end
      end
   end
  
   self.classIndices = {}
   for k,v in ipairs(self.classes) do
      self.classIndices[v] = k
   end

   -- define command-line tools, try your best to maintain OSX compatibility
   local wc = 'wc'
   local cut = 'cut'
   local find = 'find'
   if jit.os == 'OSX' then
      wc = 'gwc'
      cut = 'gcut'
      find = 'gfind'
   end
   ----------------------------------------------------------------------
   -- Options for the GNU find command
   -- 支持的图片格式后缀
   local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'  -- iname "*.jpg"
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end
   -- -iname "*.jpg" -o -iname "*.png" ..  
   -- find the image path names
   self.imagePath = torch.CharTensor()  -- path to each image in dataset
   self.imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)
   self.classList = {}                  -- index of imageList to each image of a particular class
   self.classListSample = self.classList -- the main list used when sampling data

   print('running "find" on each class directory, and concatenate all'
         .. ' those filenames into a single file containing all image paths for a given class')
   -- so, generates one file per class
   local classFindFiles = {}
   for i=1,#self.classes do
      classFindFiles[i] = os.tmpname()
   end
   local combinedFindList = os.tmpname();

   local tmpfile = os.tmpname()
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- iterate over classes
   for i, class in ipairs(self.classes) do
      -- iterate over classPaths
      for j,path in ipairs(classPaths[i]) do
         local command = find .. ' "' .. path .. '" ' .. findOptions
            .. ' >>"' .. classFindFiles[i] .. '" \n'
         tmphandle:write(command)
      end
   end
   io.close(tmphandle)
   os.execute('bash ' .. tmpfile)
   os.execute('rm -f ' .. tmpfile)

   print('now combine all the files to a single large file')
   local tmpfile = os.tmpname()
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- concat all finds to a single large file in the order of self.classes
   for i=1,#self.classes do
      local command = 'cat "' .. classFindFiles[i] .. '" >>' .. combinedFindList .. ' \n'
      tmphandle:write(command)
   end
   io.close(tmphandle)
   os.execute('bash ' .. tmpfile)
   os.execute('rm -f ' .. tmpfile)

   --==========================================================================
   print('load the large concatenated list of sample paths to self.imagePath')
   local maxPathLength = tonumber(sys.fexecute(wc .. " -L '"
                                                  .. combinedFindList .. "' |"
                                                  .. cut .. " -f1 -d' '")) + 1
   local length = tonumber(sys.fexecute(wc .. " -l '"
                                           .. combinedFindList .. "' |"
                                           .. cut .. " -f1 -d' '"))
   assert(length > 0, "Could not find any image file in the given input paths")
   assert(maxPathLength > 0, "paths of files are length 0?")
   self.imagePath:resize(length, maxPathLength):fill(0)
   local s_data = self.imagePath:data()
   local count = 0
   for line in io.lines(combinedFindList) do
      ffi.copy(s_data, line)
      s_data = s_data + maxPathLength
      if self.verbose and count % 10000 == 0 then
         xlua.progress(count, length)
      end;
      count = count + 1
   end

   self.numSamples = self.imagePath:size(1)
   if self.verbose then print(self.numSamples ..  ' samples found.') end
   --==========================================================================
   print('Updating classList and imageClass appropriately')
   self.imageClass:resize(self.numSamples)
   local runningIndex = 0
   for i=1,#self.classes do
      if self.verbose then xlua.progress(i, #(self.classes)) end
      local length = tonumber(sys.fexecute(wc .. " -l '"
                                              .. classFindFiles[i] .. "' |"
                                              .. cut .. " -f1 -d' '"))
      if length == 0 then
         error('Class has zero samples')
      else
         self.classList[i] = torch.linspace(runningIndex + 1, runningIndex + length, length):long()
         self.imageClass[{{runningIndex + 1, runningIndex + length}}]:fill(i)
      end
      runningIndex = runningIndex + length
   end

   --==========================================================================
   -- clean up temporary files
   print('Cleaning up temporary files')
   local tmpfilelistall = ''
   for i=1,#(classFindFiles) do
      tmpfilelistall = tmpfilelistall .. ' "' .. classFindFiles[i] .. '"'
      if i % 1000 == 0 then
         os.execute('rm -f ' .. tmpfilelistall)
         tmpfilelistall = ''
      end
   end
   os.execute('rm -f '  .. tmpfilelistall)
   os.execute('rm -f "' .. combinedFindList .. '"')
   --==========================================================================

   if self.split == 100 then
      self.testIndicesSize = 0
   else
      print('Splitting training and test sets to a ratio of '
               .. self.split .. '/' .. (100-self.split))
      self.classListTrain = {}
      self.classListTest  = {}
      self.classListSample = self.classListTrain
      local totalTestSamples = 0
      -- split the classList into classListTrain and classListTest
      for i=1,#self.classes do
         local list = self.classList[i]
         local count = self.classList[i]:size(1)
         local splitidx = math.floor((count * self.split / 100) + 0.5) -- +round
         local perm = torch.randperm(count)
         self.classListTrain[i] = torch.LongTensor(splitidx)
         for j=1,splitidx do
            self.classListTrain[i][j] = list[perm[j]]
         end
         if splitidx == count then -- all samples were allocated to train set
            self.classListTest[i]  = torch.LongTensor()
         else
            self.classListTest[i]  = torch.LongTensor(count-splitidx)
            totalTestSamples = totalTestSamples + self.classListTest[i]:size(1)
            local idx = 1
            for j=splitidx+1,count do
               self.classListTest[i][idx] = list[perm[j]]
               idx = idx + 1
            end
         end
      end
      -- Now combine classListTest into a single tensor
      self.testIndices = torch.LongTensor(totalTestSamples)
      self.testIndicesSize = totalTestSamples
      local tdata = self.testIndices:data()
      local tidx = 0
      for i=1,#self.classes do
         local list = self.classListTest[i]
         if list:dim() ~= 0 then
            local ldata = list:data()
            for j=0,list:size(1)-1 do
               tdata[tidx] = ldata[j]
               tidx = tidx + 1
            end
         end
      end
   end
end

-- size(), size(class)
function dataset:size(class, list)  -- 给定class的索引或者类名称， 返回该类具有的数据总数
   list = list or self.classList
   if not class then
      return self.numSamples
   elseif type(class) == 'string' then
      return list[self.classIndices[class]]:size(1)
   elseif type(class) == 'number' then
      return list[class]:size(1)
   end
end

-- getByClass
-- 根据class的类别选取随机选取该类训练样本的一个样本
function dataset:getByClass(class)
   -- index是classListSample[class]中的一个要被选择的随机样本的索引值
   local index = math.max(1, math.ceil(torch.uniform() * self.classListSample[class]:nElement()))
   -- 获取被选中的该样本的虽在的路径
   local imgpath = ffi.string(torch.data(self.imagePath[self.classListSample[class][index]]))
   --  返回该样本， 这里调用了加载该随机样本的函数：sampleHookTrain函数
   return self:sampleHookTrain(imgpath)
end

-- converts a table of samples (and corresponding labels) to a clean tensor
-- 将table中的数据存放到一个大的tensor中， 方便处理
local function tableToOutput(self, dataTable, scalarTable)
   local data, scalarLabels, labels
   local quantity = #scalarTable
   assert(dataTable[1]:dim() == 3)
   -- data 存放着传入table的所有样本紧缩后的大的tensor空间
   data = torch.Tensor(quantity,
		       self.sampleSize[1], self.sampleSize[2], self.sampleSize[3])
   -- 数据标签存放在新的大的tensor中（size为quantity）
   scalarLabels = torch.LongTensor(quantity):fill(-1111)
   for i=1,#dataTable do
      data[i]:copy(dataTable[i])
      scalarLabels[i] = scalarTable[i]
   end
   return data, scalarLabels
end

-- sampler, samples from the training set.
function dataset:sample(quantity)
   assert(quantity)
   local dataTable = {}
   local scalarTable = {}
   -- 随机采样quantity数目个样本
   for i=1,quantity do
      -- 确定要选择样本的随机类别
      local class = torch.random(1, #self.classes)
      -- 根据该随机类别， 选择该类别下的样本
      local out = self:getByClass(class)
      -- 将该类别的一个随机样本存到数据table中
      table.insert(dataTable, out)
      -- 存储该样本的类别型号到另一个table中
      table.insert(scalarTable, class)
   end
    -- 函数调用之后返回的数据和标签数据：
   -- data：data = torch.Tensor(quantity,self.sampleSize[1], self.sampleSize[2], self.sampleSize[3])
   -- scalarLablels: torch.LongTensor(quantity)
   local data, scalarLabels = tableToOutput(self, dataTable, scalarTable)
   return data, scalarLabels
end

-- 获取索引范围在[i1, i2]之间的所有样本
function dataset:get(i1, i2)
   local indices = torch.range(i1, i2);
   local quantity = i2 - i1 + 1;
   assert(quantity > 0)
   -- now that indices has been initialized, get the samples
   local dataTable = {}
   local scalarTable = {}
   for i=1,quantity do
      -- load the sample
      local imgpath = ffi.string(torch.data(self.imagePath[indices[i]]))
      local out = self:sampleHookTest(imgpath)
      table.insert(dataTable, out)
      table.insert(scalarTable, self.imageClass[indices[i]])
   end
   -- 函数调用之后返回的数据和标签数据：
   -- data：data = torch.Tensor(quantity,self.sampleSize[1], self.sampleSize[2], self.sampleSize[3])
   -- scalarLablels: torch.LongTensor(quantity)
   local data, scalarLabels = tableToOutput(self, dataTable, scalarTable)
   -- 返回batch大小为为quantity的数据和标签
   return data, scalarLabels
end

return dataset
